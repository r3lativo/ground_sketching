# src/pipeline.py

import asyncio
import logging
import random
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

from src.utils import (
    pil_to_base64,
    base64_to_pil,
    clean_image_artifacts,
    load_existing_image,
    process_zoom,
    process_generated_image,
    is_not_empty_val,
    mock_creation_logic,
    mock_gen_logic
)
from src.api_clients import Action

# Note: We rely on the TaskLogger passed from augmenter, not the global logger
global_logger = logging.getLogger(__name__)

class AugmentationPipeline:
    def __init__(self, api_client, mock_mode: bool = False):
        self.client = api_client
        self.mock_mode = mock_mode

    async def run_single_user(
        self,
        data_manager,
        user: str,
        global_semaphore: asyncio.Semaphore,
        task_logger,
        pipeline_config: dict
    ):
        """
        Runs the pipeline for ONE specific user in a specific file.
        """
        task_logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        # Unpack Config
        create = pipeline_config.get('create', False)
        render = pipeline_config.get('render', False)
        oracle = pipeline_config.get('oracle', False)
        img_out_dir = pipeline_config.get('img_output_dir')

        # Setup Render Path
        user_out_path = None
        if render and img_out_dir:
            user_out_path = Path(img_out_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        sorted_indices = sorted(data_manager.df.index.tolist())
        current_image_b64 = None
        
        # State tracking
        state = {
            'frame_idx': 0,
            'seq_idx': 1,
            'current_image_b64': None,
            'prev_image_path': None
        }

        # Initialize Context (Load first image if exists)
        # In a real run, we might want to preload the very first image if we are continuing
        # For now, we assume sequential processing from start or clean slate logic per file.

        for index in sorted_indices:
            try:
                row = data_manager.df.loc[index]
            except KeyError:
                continue
            
            if row['character'] != user:
                continue

            # Load existing image context if we don't have one in memory yet
            if state['current_image_b64'] is None and is_not_empty_val(row.get('img_path')):
                img_p = str(row['img_path'])
                loaded = await asyncio.to_thread(load_existing_image, img_p)
                if loaded: state['current_image_b64'] = loaded

            updates_made = False
            # Get initial index
            data_manager.set_start_idx(index, user, oracle)

            # --- PHASE 1: CREATE ---
            if create:
                created = await self._phase_create(
                    data_manager, index, user, oracle, 
                    state['current_image_b64'], 
                    global_semaphore, task_logger
                )
                if created: updates_made = True

            # --- PHASE 2: RENDER ---
            if render and user_out_path:
                rendered = await self._phase_render(
                    data_manager, index, user, oracle, 
                    state, user_out_path, 
                    global_semaphore, task_logger
                )
                if rendered: updates_made = True

            # Save periodically (using the thread-safe lock in DataManager)
            if updates_made:
                await data_manager.save()

        task_logger.info(f"--- Finished User: {user} ---")

    async def _phase_create(self, dm, index, user, oracle, current_img, sem, logger):
        """
        Handles Logic: VLM Decision -> Strategy Execution -> Prompt Update
        """
        row = dm.df.loc[index]
        utterance = f"{row['character']}: {row['text']}"
        context = dm.get_context_for_index(index, user)
        prev_prompts = dm.get_prev_prompts_for_index(index, user, oracle)

        # 1. Get Strategy (Throttled)
        async with sem:
            logger.info(f"[CREATE] Index {index}: Asking VLM for strategy...")
            decision = await self.client.get_meta_and_strategy(
                is_oracle=oracle,
                utterance=utterance,
                context=context,
                previous_prompts=prev_prompts,
                has_images=(current_img is not None)
            )

        # 2. Log Trace
        logger.log_trace(index, "strategy_decision", decision.to_dict())

        # 3. Handle SKIP
        if not decision.action or decision.action == Action.SKIP:
            logger.info(f"[CREATE] Index {index}: Action is SKIP.")
            await dm.update_cell(index, 'frame_choice', Action.SKIP)
            return False

        # 4. Handle NEW vs KEEP
        if decision.action == Action.NEW:
            prev_prompts = [] # Reset prompt context for the strategy generation
        
        if decision.imagery_utterance:
            utterance = decision.imagery_utterance

        # Write Metadata
        await dm.update_cell(index, 'frame_choice', decision.action)
        await dm.update_cell(index, 'meta_info', decision.meta_info)
        await dm.update_cell(index, 'initial_prompt', decision.imagery_utterance)

        # 5. Execute Strategy (Throttled)
        new_prompt = ""
        if self.mock_mode:
            new_prompt = mock_creation_logic(utterance)
        elif decision.strategy:
            async with sem:
                logger.info(f"[CREATE] Index {index}: Executing Strategy {decision.strategy_name}...")
                new_prompt = await self.client.execute_vlm_strategy(
                    decision.strategy, utterance, context, prev_prompts
                )
                logger.log_trace(index, "strategy_generation", {"prompt": new_prompt})

        await dm.update_cell(index, 'initial_prompt', new_prompt)
        return True

    async def _phase_render(self, dm, index, user, oracle, state, out_path, sem, logger):
        """
        Handles Logic: Refinement -> Image Gen -> Save -> Update State
        """
        row = dm.df.loc[index]
        initial_prompt = str(row.get('initial_prompt', '')).strip()
        
        if not is_not_empty_val(initial_prompt):
            return False

        # 1. Refine Prompt (Throttled)
        final_prompt = initial_prompt
        if state['current_image_b64']:
            if self.mock_mode:
                final_prompt = f"[Refined] {initial_prompt}"
            else:
                async with sem:
                    mm_decision = await self.client.get_meta_and_strategy(is_oracle=oracle, has_images=True)
                    if mm_decision.strategy:
                        logger.info(f"[RENDER] Index {index}: Refining prompt with visual context...")
                        refined = await self.client.execute_vlm_strategy(
                            mm_decision.strategy, initial_prompt, images=[state['current_image_b64']]
                        )
                        if refined: final_prompt = refined
                        logger.log_trace(index, "refinement", {"before": initial_prompt, "after": final_prompt})

        await dm.update_cell(index, 'final_prompt', final_prompt)

        # 2. Generate Images
        sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]
        updates = False

        for sub_prompt in sub_prompts:
            img_filename = f"{user}_{state['frame_idx']}_seq{state['seq_idx']}.png"
            save_file = out_path / img_filename
            state['seq_idx'] += 1

            # Handle Zoom
            if "[ZOOM_OUT]" in sub_prompt.upper():
                if state['current_image_b64']:
                    # CPU bound, use thread
                    new_b64 = await asyncio.to_thread(process_zoom, state['current_image_b64'], save_file)
                    if new_b64:
                        state['current_image_b64'] = new_b64
                        state['prev_image_path'] = str(save_file)
                continue

            # Handle Gen (Throttled)
            async with sem:
                logger.info(f"[RENDER] Index {index}: Generating image...")
                if self.mock_mode:
                    new_b64_raw = mock_gen_logic(sub_prompt)
                else:
                    imgs_payload = [state['current_image_b64']] if state['current_image_b64'] else None
                    new_b64_raw = await self.client.call_image_gen(sub_prompt, imgs_payload)

            if new_b64_raw:
                # Save/Clean (CPU bound)
                processed_b64 = await asyncio.to_thread(process_generated_image, new_b64_raw, save_file)
                if processed_b64:
                    state['current_image_b64'] = processed_b64
                    state['prev_image_path'] = str(save_file)

        if state['prev_image_path']:
             await dm.update_cell(index, 'img_path', state['prev_image_path'])
             updates = True

        return updates
