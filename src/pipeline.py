# src/pipeline.py

import asyncio
import logging
import random
import os
import pandas as pd
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw

from src.utils import pil_to_base64, base64_to_pil, clean_image_artifacts, add_padding_to_image
from src.data_manager import ConversationDataManager
from src.api_clients import (
    execute_vlm_strategy,
    get_meta_and_strategy,
    call_image_gen
)

logger = logging.getLogger(__name__)

class AugmentationPipeline:
    """
    Orchestrates the augmentation process.
    Handles parallel execution of users and pipelining of Create -> Render stages.
    """

    def __init__(self, data_manager: ConversationDataManager, mock_mode: bool = False):
        self.dm = data_manager
        self.mock_mode = mock_mode
        self.vlm_semaphore = asyncio.Semaphore(1) # Limit concurrent VLM calls
        # TODO think about re implementing parallelization

    # --- Entry Points ---

    async def run_full_pipeline(
        self, 
        users: List[str], 
        create: bool, 
        render: bool, 
        realistic_context: bool, 
        output_dir: str,
        chunk_col_resolver: callable
    ):
        """
        Runs the pipeline for all users in parallel.
        Pipelines the Creation -> Rendering stages for each chunk.
        """
        tasks = []
        for user in users:
            chunk_col = chunk_col_resolver(user)
            tasks.append(
                self._run_user_pipeline(user, chunk_col, create, render, realistic_context, output_dir)
            )
        
        # Run all users in parallel
        await asyncio.gather(*tasks)
        
        # Final save to ensure everything is flushed
        await self.dm.save()

    async def _run_user_pipeline(self, user, chunk_col, create, render, realistic, output_dir):
        logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        user_out_path = None
        if render and output_dir:
            user_out_path = Path(output_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        chunk_tasks = []
        
        # Group by chunk and launch a chain for each
        for chunk_id, chunk_df in self.dm.get_user_groups(user, chunk_col):
            # We pass the INDICES, not the dataframe slice. 
            # This ensures workers pull fresh data (e.g. Render sees what Create just wrote).
            chunk_indices = chunk_df.index.tolist()
            
            chunk_tasks.append(
                self._process_chunk_chain(chunk_id, chunk_indices, user, create, render, realistic, user_out_path)
            )
            
        await asyncio.gather(*chunk_tasks)

    async def _process_chunk_chain(self, chunk_id, indices, user, create, render, realistic, output_path):
        """Sequential chain for a single chunk: Create -> Render"""
        
        # 1. Creation Stage
        if create:
            await self._process_chunk_create(chunk_id, indices, user, realistic)
        
        # 2. Rendering Stage (Runs immediately after Create finishes for this chunk)
        if render and output_path:
            await self._process_chunk_render(chunk_id, indices, user, output_path)

    # --- Worker Logic: Creation ---

    async def _process_chunk_create(self, chunk_id, indices, user, realistic):
        async with self.vlm_semaphore:
            logger.info(f"Processing Chunk {chunk_id} (Create) - {user}")
            
            # Sort indices to process dialogue in order
            sorted_indices = sorted(indices)
            previous_prompts = self.dm.get_prev_prompts_for_index(sorted_indices[0], user, realistic)
            updates_made = False

            for index in sorted_indices:
                # Fetch fresh row
                try:
                    row = self.dm.df.loc[index]
                except KeyError:
                    continue # Row might have been dropped (unlikely)

                if row['character'] != user:
                    continue
                
                # Check for existing valid prompt
                current_val = row.get('initial_prompt')
                if self._is_valid_prompt(current_val):
                    if not self._is_no_change(current_val):
                        previous_prompts.append(str(current_val).strip())
                    continue

                # Prepare Inputs
                utterance = f"{row['character']}: {row['text']}"
                context = self.dm.get_context_for_index(index, user, realistic)

                # Debug Logging
                logger.info(f"Index: {index} - Utterance: '{utterance}'")
                logger.info(f"Context: {context}")
                logger.info(f"Previous prompts: {previous_prompts}")

                # Strategy Selection
                strategy, strategy_name, meta_info, imagery_utterance = await get_meta_and_strategy(
                    is_oracle=self.mock_mode,
                    utterance=utterance,
                    context=context,
                    previous_prompts=previous_prompts
                )
                
                if strategy is None:
                    print("NO STRATEGY??")
                    return

                # Base: it's a new frame
                choice = '[NEW]'

                if 'edit' in strategy_name:
                    choice = '[CONTINUE]'

                # Wipe p_p if create (so, new frame)
                if 'create' in strategy_name:
                    previous_prompts = []
                
                # Update dataframe
                self.dm.update_cell(index, 'frame_choice', choice)
                self.dm.update_cell(index, 'meta_info', meta_info)
                self.dm.update_cell(index, 'imagery_utterance', imagery_utterance)

                # Update the utterance to the new utterance subtracted of the meta information.
                if meta_info is not None:
                    utterance = imagery_utterance

                # Execute
                if self.mock_mode:
                    new_prompt = self._mock_creation_logic(utterance)
                else:
                    new_prompt = await execute_vlm_strategy(
                        strategy, utterance, context, previous_prompts
                    )

                # Update Data
                if new_prompt:
                    clean_p = "[NO_CHANGE]" if self._is_no_change(new_prompt) else new_prompt
                    self.dm.update_cell(index, 'initial_prompt', clean_p)
                    updates_made = True
                    
                    if not self._is_no_change(clean_p):
                        previous_prompts.append(clean_p)
                else:
                    self.dm.update_cell(index, 'initial_prompt', "")
                    updates_made = True

            if updates_made:
                await self.dm.save()

    # --- Worker Logic: Rendering ---

    async def _process_chunk_render(self, chunk_id, indices, user, output_path: Path):
        logger.info(f"Processing Chunk {chunk_id} (Render) - {user}")
        
        current_image_b64 = None
        sorted_indices = sorted(indices)
        updates_made = False

        for index in sorted_indices:
            # Fetch fresh row (Crucial: Create might have just updated 'initial_prompt')
            try:
                row = self.dm.df.loc[index]
            except KeyError:
                continue

            if row['character'] != user or not self._is_valid_prompt(row.get('initial_prompt')):
                continue

            # Load existing image if available (continuity)
            if self._is_valid_prompt(row.get('img_path')) and os.path.exists(row['img_path']):
                try:
                    with Image.open(row['img_path']) as img:
                        current_image_b64 = pil_to_base64(img)
                    continue
                except:
                    pass

            initial_prompt = str(row['initial_prompt']).strip()
            choice = str(row['frame_choice'])

            # If NEW, wipe out current image
            if choice == '[NEW]':
                current_image_b64 = None
            
            if self._is_no_change(initial_prompt):
                # Ensure marker is set if missing
                if row.get('initial_prompt') != "[NO_CHANGE]":
                    self.dm.update_cell(index, 'initial_prompt', "[NO_CHANGE]")
                    updates_made = True
                continue

            # 1. Refine Prompt
            final_prompt = initial_prompt
            if current_image_b64:
                if self.mock_mode:
                     final_prompt = f"[Refined] {initial_prompt}"
                else:
                    # Stage 2 Strategy: Multimodal Edit
                    strategy, strategy_name = await get_meta_and_strategy(
                        is_oracle=False,
                        has_images=True
                    )
                    try:
                        async with self.vlm_semaphore:
                            refined = await execute_vlm_strategy(
                                strategy, initial_prompt, images=[current_image_b64]
                            )
                        if refined and not self._is_no_change(refined):
                            final_prompt = refined
                    except Exception as e:
                        logger.error(f"Refine failed at {index}: {e}")

            # SAFE CHECK FOR PD.NA
            if self._safe_value_changed(row.get('final_prompt'), final_prompt):
                self.dm.update_cell(index, 'final_prompt', final_prompt)
                updates_made = True

            # 2. Render
            sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]
            last_saved_path = None

            for step_i, sub_prompt in enumerate(sub_prompts):
                suffix = f"_seq{step_i}" if len(sub_prompts) > 1 else ""
                img_filename = f"chunk_{chunk_id}_{user}_{index}{suffix}.png"
                save_file = output_path / img_filename

                # ZOOM Logic
                if "[ZOOM_OUT]" in sub_prompt.upper():
                    if current_image_b64:
                        try:
                            pil_img = base64_to_pil(current_image_b64)
                            zoomed = add_padding_to_image(pil_img, scale_factor=0.8)
                            zoomed.save(save_file)
                            current_image_b64 = pil_to_base64(zoomed)
                            last_saved_path = str(save_file)
                        except Exception as e:
                            logger.error(f"Zoom error: {e}")
                    continue
                
                # GEN Logic
                if self.mock_mode:
                    new_b64 = self._mock_gen_logic(sub_prompt)
                else:
                    new_b64 = await call_image_gen(
                        sub_prompt, 
                        [current_image_b64] if current_image_b64 else None
                    )

                if new_b64:
                    try:
                        img = clean_image_artifacts(base64_to_pil(new_b64))
                        img.save(save_file)
                        current_image_b64 = pil_to_base64(img)
                        last_saved_path = str(save_file)
                    except Exception as e:
                        logger.error(f"Save error: {e}")

            # SAFE CHECK FOR PD.NA
            if last_saved_path and self._safe_value_changed(row.get('img_path'), last_saved_path):
                self.dm.update_cell(index, 'img_path', last_saved_path)
                updates_made = True
        
        if updates_made:
            await self.dm.save()

    # --- Helpers & Mocks ---

    def _safe_value_changed(self, old_val, new_val):
        """Safely checks if a value has changed, handling pd.NA/NaN."""
        # Treat pd.NA / np.nan / None as effectively the same "missing" state
        is_old_missing = pd.isna(old_val)
        is_new_missing = pd.isna(new_val)

        if is_old_missing and is_new_missing:
            return False
        if is_old_missing != is_new_missing:
            return True
        return str(old_val) != str(new_val)

    def _is_valid_prompt(self, val):
        """Safe check for non-empty, non-NA prompt strings."""
        if pd.isna(val):
            return False
        return str(val).strip() != ""

    def _is_no_change(self, val):
        """Safe check for NO_CHANGE token."""
        if pd.isna(val):
            return False
        return str(val).strip().upper() in ("[NO_CHANGE]", "NO_CHANGE", "[NO CHANGE]")

    def _mock_creation_logic(self, utterance):
        """Generates a mock prompt based on random logic."""
        val = random.random()
        if val < 0.1: return "[NO_CHANGE]"
        elif val < 0.3: return f"Close up of {utterance} $$$ [ZOOM_OUT] $$$ Wide of {utterance}"
        elif val < 0.5: return f"First angle {utterance} $$$ Second angle {utterance}"
        return f"[Mock Prompt] {utterance}"

    def _mock_gen_logic(self, prompt):
        """Generates a random colored square."""
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        img = Image.new('RGB', (128, 128), color=color)
        d = ImageDraw.Draw(img)
        d.rectangle([10,10,40,40], fill="white")
        if "[ZOOM_OUT]" in str(prompt): d.text((10,50), "ZOOM", fill="white")
        return pil_to_base64(img)
