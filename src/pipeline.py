# src/pipeline.py

import asyncio
import logging
import random
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw

from src.utils import pil_to_base64, base64_to_pil, clean_image_artifacts, add_padding_to_image
from src.data_manager import ConversationDataManager
from src.api_clients import APIClient, Action

logger = logging.getLogger(__name__)

# --- Helper Functions for ThreadPoolExecutor ---
def _io_load_existing_image(path: str) -> Optional[str]:
    """Blocking IO: Opens image from disk and converts to base64."""
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                return pil_to_base64(img)
        except Exception as e:
            logger.warning(f"Could not load existing image at {path}: {e}")
    return None

def _io_process_zoom(current_b64: str, save_path: Path) -> Optional[str]:
    """Blocking CPU/IO: Decodes, Zooms (Resizes), and Saves."""
    try:
        pil_img = base64_to_pil(current_b64)
        zoomed = add_padding_to_image(pil_img, scale_factor=0.8)
        zoomed.save(save_path)
        return pil_to_base64(zoomed)
    except Exception as e:
        logger.error(f"Zoom error: {e}")
        return None

def _io_process_generated_image(new_b64: str, save_path: Path) -> Optional[str]:
    """Blocking CPU/IO: Decodes, Cleans Artifacts, and Saves."""
    try:
        img = clean_image_artifacts(base64_to_pil(new_b64))
        img.save(save_path)
        return pil_to_base64(img)
    except Exception as e:
        logger.error(f"Save error: {e}")
        return None

# -----------------------------------------------

class AugmentationPipeline:
    """
    Orchestrates the augmentation process using APIClient.
    """

    def __init__(self, data_manager: ConversationDataManager, api_client: APIClient, mock_mode: bool = False):
        self.dm = data_manager
        self.client = api_client
        self.mock_mode = mock_mode
        self.vlm_semaphore = asyncio.Semaphore(1)

    async def run_full_pipeline(
        self, 
        users: List[str], 
        create: bool, 
        render: bool, 
        oracle: bool, 
        output_dir: str,
        chunk_col_resolver: callable = None
    ):
        """
        Runs the pipeline for all users in parallel.
        Initializes the API Client session here.
        """
        # --- Initialize API Client Session ---
        async with self.client:
            tasks = []
            for user in users:
                tasks.append(
                    self._run_user_pipeline(user, create, render, oracle, output_dir)
                )
            await asyncio.gather(*tasks)
            
            # Final save to ensure everything is flushed at the end of the run
            await self.dm.save()

    async def _run_user_pipeline(self, user, create, render, oracle, output_dir):
        logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        loop = asyncio.get_running_loop()

        user_out_path = None
        if render and output_dir:
            user_out_path = Path(output_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        sorted_indices = sorted(self.dm.df.index.tolist())
        current_image_b64 = None
        
        frame_idx = 0
        seq_idx = 1

        # Batch Saving Configuration
        SAVE_INTERVAL = 1
        unsaved_changes = 0

        for index in sorted_indices:
            try: row = self.dm.df.loc[index]
            except KeyError: continue 
            if row['character'] != user: continue

            # Load previous image if it exists for Context
            if current_image_b64 is None and self._is_not_empty_val(row.get('img_path')):
                img_p = str(row['img_path'])
                current_image_b64 = await loop.run_in_executor(None, _io_load_existing_image, img_p)

            updates_made = False

            # Get initial index
            self.dm.set_start_idx(index, user, oracle)

            # --- PHASE 1: CREATE ---
            if create:
                utterance = f"{row['character']}: {row['text']}"
                context = self.dm.get_context_for_index(index, user)
                previous_prompts = self.dm.get_prev_prompts_for_index(index, user, oracle)

                # DEBUG LOGGING
                logger.info(f"User and Index: {user}, {index}")
                logger.info(f"Context: {context}")
                logger.info(f"Utterance: '{utterance}'")
                logger.info(f"Previous prompts: {previous_prompts}")

                # A. Get Meta & Strategy Decision
                async with self.vlm_semaphore:
                    logger.info("CREATE - Get Meta and Strategy...")
                    # Returns StrategyDecision object now
                    decision = await self.client.get_meta_and_strategy(
                        is_oracle=oracle,
                        utterance=utterance,
                        context=context,
                        previous_prompts=previous_prompts,
                        has_images=False
                    )

                # B. Handle SKIP Logic
                # If decision object is empty or Action is SKIP
                if not decision.action or decision.action == Action.SKIP:
                    logger.info(f"Skipping index {index} based on VLM decision: {decision.action}")
                    
                    # We still record that we looked at it and decided to skip
                    await self.dm.update_cell(index, 'frame_choice', decision.action or Action.SKIP)
                    await self.dm.update_cell(index, 'meta_info', decision.meta_info)
                    await self.dm.update_cell(index, 'initial_prompt', decision.imagery_utterance)
                    continue

                # C. Handle Strategy Selection & Meta Substitution
                if decision.action == Action.NEW:
                    previous_prompts = []
                    current_image_b64 = None

                if decision.imagery_utterance:
                    logger.info(f"Using extracted imagery utterance for generation.")
                    utterance = decision.imagery_utterance
                else:
                    # If no imagery utterance but NOT skipped, we just use the raw utterance
                    pass

                # D. Update DataFrame with Decision Data
                await self.dm.update_cell(index, 'frame_choice', decision.action)
                await self.dm.update_cell(index, 'meta_info', decision.meta_info)
                await self.dm.update_cell(index, 'imagery_utterance', decision.imagery_utterance)

                # E. Execute Creation Strategy
                new_prompt = ""
                if self.mock_mode:
                    new_prompt = self._mock_creation_logic(utterance)
                elif decision.strategy:
                    async with self.vlm_semaphore:
                        logger.info(f"CREATE - Execute Strategy {decision.strategy_name}...")
                        new_prompt = await self.client.execute_vlm_strategy(
                            decision.strategy, utterance, context, previous_prompts
                        )

                # F. Save Initial Prompt
                await self.dm.update_cell(index, 'initial_prompt', new_prompt)
                
                updates_made = True

            # --- PHASE 2: RENDER ---
            if render and user_out_path:
                # Re-fetch row in case CREATE just updated it
                row = self.dm.df.loc[index]
                initial_prompt = str(row.get('initial_prompt', '')).strip()

                # Basic validation: If empty, we can't render
                if not self._is_not_empty_val(initial_prompt): continue

                # A. Refine Prompt
                final_prompt = initial_prompt
                if current_image_b64:
                    if self.mock_mode:
                        final_prompt = f"[Refined] {initial_prompt}"
                    else:
                        # 1. Get MM Strategy (Forces MultimodalEditStrategy)
                        mm_decision = await self.client.get_meta_and_strategy(
                            is_oracle=oracle, has_images=True
                        )
                        
                        # 2. Execute Refinement
                        if mm_decision.strategy:
                            async with self.vlm_semaphore:
                                logger.info(f"RENDER - Execute Multimodal Edit...")
                                refined = await self.client.execute_vlm_strategy(
                                    mm_decision.strategy, initial_prompt, images=[current_image_b64]
                                )
                            
                            if refined:
                                final_prompt = refined
                else:
                    # If NO CURRENT IMG then update frame and reset sequence
                    if seq_idx > 1:
                        frame_idx += 1
                    seq_idx = 1
                    
                await self.dm.update_cell(index, 'final_prompt', final_prompt)
                updates_made = True

                # B. Image Generation Loop
                sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]
                last_saved_path = None

                for sub_prompt in sub_prompts:
                    img_filename = f"{user}_{frame_idx}_seq{seq_idx}.png"
                    save_file = user_out_path / img_filename
                    seq_idx += 1

                    # Handle ZOOM_OUT logic
                    if "[ZOOM_OUT]" in sub_prompt.upper():
                        if current_image_b64:
                            new_b64 = await loop.run_in_executor(
                                None, _io_process_zoom, current_image_b64, save_file
                            )
                            if new_b64:
                                current_image_b64 = new_b64
                                last_saved_path = str(save_file)
                        continue
                    
                    # Generate New Image
                    imgs_payload = [current_image_b64] if current_image_b64 else None
                    new_b64_raw = None

                    if self.mock_mode:
                        new_b64_raw = self._mock_gen_logic(sub_prompt)
                    else:
                        new_b64_raw = await self.client.call_image_gen(sub_prompt, imgs_payload)

                    if new_b64_raw:
                        processed_b64 = await loop.run_in_executor(
                            None, _io_process_generated_image, new_b64_raw, save_file
                        )
                        if processed_b64:
                            current_image_b64 = processed_b64
                            last_saved_path = str(save_file)

                if last_saved_path:
                    await self.dm.update_cell(index, 'img_path', last_saved_path)
                    updates_made = True

            # --- BATCH SAVING LOGIC ---
            if updates_made:
                unsaved_changes += 1
                if unsaved_changes >= SAVE_INTERVAL:
                    logger.info(f"Saving progress for user {user}...")
                    await self.dm.save()
                    unsaved_changes = 0

        # Ensure final changes are saved when user loop finishes
        if unsaved_changes > 0:
            await self.dm.save()
        logger.info(f"--- Finished Pipeline for User: {user} ---")

    # --- Helpers & Mocks ---

    def _is_not_empty_val(self, val):
        if pd.isna(val): return False
        return str(val).strip() != ""

    def _mock_creation_logic(self, utterance):
        val = random.random()
        if val < 0.3: return f"Close up of {utterance} $$$ [ZOOM_OUT] $$$ Wide of {utterance}"
        elif val < 0.6: return f"First angle {utterance} $$$ Second angle {utterance}"
        return f"[Mock Prompt] {utterance}"

    def _mock_gen_logic(self, prompt):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        img = Image.new('RGB', (128, 128), color=color)
        d = ImageDraw.Draw(img)
        d.rectangle([10,10,40,40], fill="white")
        if "[ZOOM_OUT]" in str(prompt): d.text((10,50), "ZOOM", fill="white")
        return pil_to_base64(img)
