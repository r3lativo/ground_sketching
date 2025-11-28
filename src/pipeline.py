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
from src.api_clients import APIClient

logger = logging.getLogger(__name__)

# --- Helper Functions for ThreadPoolExecutor ---
# These run in a separate thread to prevent blocking the async event loop.

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
        realistic_context: bool, 
        output_dir: str,
        chunk_col_resolver: callable = None # Unused now, kept for signature comp
    ):
        """
        Runs the pipeline for all users in parallel.
        """
        tasks = []
        for user in users:
            tasks.append(
                self._run_user_pipeline(user, create, render, realistic_context, output_dir)
            )
        await asyncio.gather(*tasks)
        await self.dm.save()

    async def _run_user_pipeline(self, user, create, render, realistic, output_dir):
        logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        # Get the running loop for executor calls
        loop = asyncio.get_running_loop()

        user_out_path = None
        if render and output_dir:
            user_out_path = Path(output_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        sorted_indices = sorted(self.dm.df.index.tolist())
        current_image_b64 = None 
        
        for index in sorted_indices:
            try: row = self.dm.df.loc[index]
            except KeyError: continue 
            if row['character'] != user: continue

            if current_image_b64 is None and self._is_not_empty_val(row.get('img_path')):
                img_p = str(row['img_path'])
                current_image_b64 = await loop.run_in_executor(None, _io_load_existing_image, img_p)

            # --- PHASE 1: CREATE ---
            if create:
                utterance = f"{row['character']}: {row['text']}"
                context = self.dm.get_context_for_index(index, user, realistic)
                previous_prompts = self.dm.get_prev_prompts_for_index(index, user, realistic)

                # A. Get Meta & Strategy
                async with self.vlm_semaphore:
                    strategy, strategy_name, meta_info, imagery_utterance = await self.client.get_meta_and_strategy(
                        is_oracle=self.mock_mode,
                        utterance=utterance,
                        context=context,
                        previous_prompts=previous_prompts,
                        has_images=(current_image_b64 is not None)
                    )

                if strategy is None:
                    continue

                # B. Handle Choices (NEW vs CONTINUE)
                choice = '[CONTINUE]'
                if 'create' in strategy_name or 'simple' in strategy_name:
                    choice = '[NEW]'
                    previous_prompts = []
                    current_image_b64 = None 

                # C. Handle Meta Substitution
                if meta_info and imagery_utterance:
                    utterance = imagery_utterance

                # D. Update DataFrame with Meta Decisions
                self.dm.update_cell(index, 'frame_choice', choice)
                self.dm.update_cell(index, 'meta_info', meta_info)
                self.dm.update_cell(index, 'imagery_utterance', imagery_utterance)

                # E. Execute Creation Strategy (Generate Initial Prompt)
                if self.mock_mode:
                    new_prompt = self._mock_creation_logic(utterance)
                else:
                    async with self.vlm_semaphore:
                        new_prompt = await self.client.execute_vlm_strategy(
                            strategy, utterance, context, previous_prompts
                        )

                # F. Save Initial Prompt
                if new_prompt:
                    clean_p = "[NO_CHANGE]" if self._is_no_change(new_prompt) else new_prompt
                    self.dm.update_cell(index, 'initial_prompt', clean_p)
                else:
                    self.dm.update_cell(index, 'initial_prompt', "")
                
                await self.dm.save()

            # --- PHASE 2: RENDER ---
            if render and user_out_path:
                # Re-fetch row in case CREATE just updated it
                row = self.dm.df.loc[index]
                initial_prompt = str(row.get('initial_prompt', '')).strip()

                if not self._is_not_empty_val(initial_prompt): continue
                if self._is_no_change(initial_prompt): continue

                # A. Refine Prompt (Multimodal Edit)
                final_prompt = initial_prompt
                if current_image_b64:
                    if self.mock_mode:
                        final_prompt = f"[Refined] {initial_prompt}"
                    else:
                        mm_strategy, _ = await self.client.get_meta_and_strategy(
                            is_oracle=False, has_images=True
                        )
                        async with self.vlm_semaphore:
                            refined = await self.client.execute_vlm_strategy(
                                mm_strategy, initial_prompt, images=[current_image_b64]
                            )
                        if refined and not self._is_no_change(refined):
                            final_prompt = refined

                self.dm.update_cell(index, 'final_prompt', final_prompt)

                # B. Image Generation Loop
                sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]
                last_saved_path = None

                for step_i, sub_prompt in enumerate(sub_prompts):
                    suffix = f"_seq{step_i}" if len(sub_prompts) > 1 else ""
                    img_filename = f"{user}_{index:05d}{suffix}.png"
                    save_file = user_out_path / img_filename

                    if "[ZOOM_OUT]" in sub_prompt.upper():
                        if current_image_b64:
                            new_b64 = await loop.run_in_executor(
                                None, _io_process_zoom, current_image_b64, save_file
                            )
                            if new_b64:
                                current_image_b64 = new_b64
                                last_saved_path = str(save_file)
                        continue
                    
                    imgs_payload = [current_image_b64] if current_image_b64 else None
                    new_b64_raw = None

                    if self.mock_mode:
                        new_b64_raw = self._mock_gen_logic(sub_prompt)
                    else:
                        # CALL VIA CLIENT INSTANCE
                        new_b64_raw = await self.client.call_image_gen(sub_prompt, imgs_payload)

                    if new_b64_raw:
                        processed_b64 = await loop.run_in_executor(
                            None, _io_process_generated_image, new_b64_raw, save_file
                        )
                        if processed_b64:
                            current_image_b64 = processed_b64
                            last_saved_path = str(save_file)

                if last_saved_path:
                    self.dm.update_cell(index, 'img_path', last_saved_path)
                await self.dm.save()

    # --- Helpers & Mocks ---

    def _is_not_empty_val(self, val):
        if pd.isna(val): return False
        return str(val).strip() != ""

    def _is_no_change(self, val):
        if pd.isna(val): return False
        return str(val).strip().upper() in ("[NO_CHANGE]", "NO_CHANGE", "[NO CHANGE]")

    def _mock_creation_logic(self, utterance):
        val = random.random()
        if val < 0.1: return "[NO_CHANGE]"
        elif val < 0.3: return f"Close up of {utterance} $$$ [ZOOM_OUT] $$$ Wide of {utterance}"
        elif val < 0.5: return f"First angle {utterance} $$$ Second angle {utterance}"
        return f"[Mock Prompt] {utterance}"

    def _mock_gen_logic(self, prompt):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        img = Image.new('RGB', (128, 128), color=color)
        d = ImageDraw.Draw(img)
        d.rectangle([10,10,40,40], fill="white")
        if "[ZOOM_OUT]" in str(prompt): d.text((10,50), "ZOOM", fill="white")
        return pil_to_base64(img)
