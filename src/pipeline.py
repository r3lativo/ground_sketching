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
    is_not_empty_val
)
from src.api_clients import Action

# Note: We rely on the TaskLogger passed from augmenter, not the global logger
global_logger = logging.getLogger(__name__)

class AugmentationPipeline:
    def __init__(self, api_client):
        self.client = api_client

    async def run_single_user(
        self,
        data_manager,
        user: str,
        global_semaphore: asyncio.Semaphore,
        t_logger,
        pipeline_config: dict
    ):
        """
        Runs the pipeline for ONE specific user in a specific file.
        """
        t_logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        # Unpack Config
        create = pipeline_config.get('create', False)
        render = pipeline_config.get('render', False)
        oracle = pipeline_config.get('oracle', False)
        candidate_count = pipeline_config.get('candidate_count')
        
        img_out_dir = pipeline_config.get('img_output_dir')

        # Setup Render Path
        user_out_path = None
        if render and img_out_dir:
            user_out_path = Path(img_out_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        sorted_indices = sorted(data_manager.df.index.tolist())
        
        # --- STATE TRACKING ---
        # frame_idx: Increments on [NEW]. Represents a distinct scene/image ID.
        # seq_idx: Increments on [CONTINUE] or sub-prompts ($$$). Represents steps within a scene.
        state = {
            'frame_idx': 0,
            'seq_idx': 1,
            'current_image_b64': None,
            'prev_image_path': None
        }

        # Context Initialization
        # If we have an existing image path in the first row we process, load it.
        # This handles resuming experiments.
        first_idx = sorted_indices[0] if sorted_indices else None
        if first_idx is not None:
             try:
                 row = data_manager.df.loc[first_idx]
                 if is_not_empty_val(row.get('img_path')):
                     img_p = str(row['img_path'])
                     loaded = await asyncio.to_thread(load_existing_image, img_p)
                     if loaded: state['current_image_b64'] = loaded
             except Exception as e:
                 t_logger.warning(f"Failed to load initial context image: {e}")

        for index in sorted_indices:
            try:
                row = data_manager.df.loc[index]
            except KeyError:
                continue
            
            if row['character'] != user:
                continue

            # Get initial index for context windowing
            data_manager.set_start_idx(index, user, oracle)

            # --- PHASE 1: CREATE ---
            if create:
                await self._phase_create(
                    data_manager, index, user, oracle,
                    global_semaphore, t_logger
                )

            # --- PHASE 2: RENDER (With Verification) ---
            if render and user_out_path:
                await self._phase_render(
                    data_manager, index, user, oracle, 
                    state, user_out_path, 
                    global_semaphore, t_logger,
                    candidate_count
                )

            # Save periodically
            await data_manager.save()

        t_logger.info(f"--- Finished User: {user} ---")

    async def _phase_create(self, dm, index, user, oracle, sem, t_logger):
        """
        Handles Logic: VLM Decision -> Strategy Execution -> Prompt Update
        """
        row = dm.df.loc[index]
        utterance = f"{row['character']}: {row['text']}"
        context = dm.get_context_for_index(index, user)
        prev_prompts = dm.get_prev_prompts_for_frame(index, user, oracle)

        # 1. Get Strategy (Throttled)
        async with sem:
            t_logger.info(f"[CREATE] Index {index}: Asking VLM for strategy...")
            # We pass has_images=False intentionally to force the VLM to focus on 
            # text creation logic rather than multimodal editing at this stage.
            decision = await self.client.get_meta_and_strategy(
                is_oracle=oracle,
                utterance=utterance,
                context=context,
                previous_prompts=prev_prompts,
                has_images=False 
            )

        # No CONTINUE without NEW
        if decision.action == Action.CONTINUE:
            # Check if this user has ever had a [NEW] frame in the past
            # We assume dm.df is the master source of truth
            history = dm.df.iloc[:index]
            
            # Fast pandas check: (User matches) AND (Frame Choice contains [NEW])
            user_has_new = history[
                (history['character'] == user) & 
                (history['frame_choice'].astype(str).str.contains(Action.NEW, regex=False))
            ].shape[0] > 0

            if not user_has_new:
                t_logger.info(f"[CREATE] Index {index}: Enforcing NEW. (Action was CONTINUE but no prior visual context found).")
                decision.action = Action.NEW

        # 2. Log Trace
        t_logger.log_trace(index, "strategy_decision", decision.to_dict())

        # 3. Handle SKIP
        if not decision.action or decision.action == Action.SKIP:
            t_logger.info(f"[CREATE] Index {index}: Action is SKIP.")
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

        # 5. Execute Strategy
        new_prompt = ""
        async with sem:
                t_logger.info(f"[CREATE] Index {index}: Executing Strategy {decision.strategy_name}...")
                new_prompt = await self.client.execute_vlm_strategy(
                    decision.strategy, utterance, context, prev_prompts
                )
                t_logger.log_trace(index, "strategy_generation", {"prompt": new_prompt})

        await dm.update_cell(index, 'initial_prompt', new_prompt)
        return True

    async def _phase_render(self, dm, index, user, oracle, state, out_path, sem, t_logger, candidate_count):
        """
        Handles Logic: State Update -> Refinement -> Best-of-N Generation -> Save
        """
        row = dm.df.loc[index]
        initial_prompt = str(row.get('initial_prompt', '')).strip()
        frame_choice = str(row.get('frame_choice', '')).strip()
        
        # --- 1. HANDLE SKIP ---
        if Action.SKIP in frame_choice:
             t_logger.info(f"[RENDER] Index {index}: Frame choice is SKIP. Skipping generation.")
             return False

        if not is_not_empty_val(initial_prompt):
            return False

        # --- 2. UPDATE STATE (Frame & Seq) ---
        is_new_frame = (Action.NEW in frame_choice)
        
        if is_new_frame:
            # NEW: Increment frame, reset sequence, clear visual context
            state['frame_idx'] += 1
            state['seq_idx'] = 1
            current_context_image = None
            t_logger.info(f"[RENDER] Index {index}: [NEW] frame detected. Starting Frame {state['frame_idx']}.")
        else:
            # CONTINUE: Keep frame, continue sequence, keep visual context
            # (seq_idx is implicitly continued from previous loop)
            current_context_image = state['current_image_b64']
            t_logger.info(f"[RENDER] Index {index}: [CONTINUE] frame detected. Continuing Frame {state['frame_idx']}, Seq {state['seq_idx']}.")

        # --- 3. REFINE PROMPT ---
        # We only refine if we have visual context (CONTINUE mode).
        # If NEW, we act as if we have no prior visual context.
        final_prompt = initial_prompt
        
        if current_context_image and not is_new_frame:
            async with sem:
                mm_decision = await self.client.get_meta_and_strategy(is_oracle=oracle, has_images=True)
                if mm_decision.strategy:
                    t_logger.info(f"[RENDER] Index {index}: Refining prompt with visual context...")
                    refined = await self.client.execute_vlm_strategy(
                        mm_decision.strategy, initial_prompt, images=[current_context_image]
                    )
                    if refined: final_prompt = refined
                    t_logger.log_trace(index, "refinement", {"before": initial_prompt, "after": final_prompt})

        await dm.update_cell(index, 'final_prompt', final_prompt)

        # --- 4. GENERATE & VERIFY LOOP ---
        sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]
        updates = False

        for sub_prompt in sub_prompts:
            img_filename = f"{user}_{state['frame_idx']}_seq{state['seq_idx']}.png"
            save_file = out_path / img_filename
            
            # Increment sequence for the *next* image (or next sub-prompt)
            state['seq_idx'] += 1

            # A. Handle Zoom (Skip Verification for Zoom)
            if "[ZOOM_OUT]" in sub_prompt.upper():
                if current_context_image:
                    new_b64 = await asyncio.to_thread(process_zoom, current_context_image, save_file)
                    if new_b64:
                        state['current_image_b64'] = new_b64 
                        state['prev_image_path'] = str(save_file)
                        current_context_image = new_b64
                continue

            # B. Prepare for Best-of-N Verification
            # We need the full prompt history to determine the "True Facts"
            prev_prompts = dm.get_prev_prompts_for_frame(index, user, oracle)
            # Add current sub_prompt to history to get complete picture
            full_history = prev_prompts + [sub_prompt]

            # Step B.1: Decompose Text into Facts (Once)
            visual_facts = []
            async with sem:
                 visual_facts = await self.client.call_prompt_summarizer(full_history)
            
            # Step B.2: Pipelined Loop
            best_candidate_b64 = None
            best_score = -1.0
            best_verification_details = []
            
            # Queue to hold the running verification task
            verification_task = None
            pending_candidate_b64 = None # The image currently being verified

            # We loop one extra time to allow the final verification to finish
            actual_loops = candidate_count if visual_facts else 1
            
            for i in range(actual_loops + 1):
                
                # --- A. CHECK PREVIOUS VERIFICATION ---
                # Before starting new work, check if the previous background task finished
                if verification_task:
                    # Wait for the VLM to finish checking the PREVIOUS image
                    score, details = await verification_task
                    
                    t_logger.info(f"[RENDER] Candidate Score: {score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_candidate_b64 = pending_candidate_b64
                        best_verification_details = details
                    
                    # Early Stopping: If perfect, we can cancel future work
                    if score >= 1.0:
                        t_logger.info(f"[RENDER] Perfect score achieved. Stopping early.")
                        break

                # --- B. STOPPING CONDITION ---
                # If we have finished generating N images, stop the loop
                if i >= actual_loops:
                    break

                # --- C. GENERATE NEXT IMAGE ---
                t_logger.info(f"[RENDER] Generating Candidate {i+1}/{actual_loops}...")

                # Generate a random seed for this specific candidate
                candidate_seed = random.randint(0, 2**32 - 1)
                
                async with sem:
                    imgs_payload = [current_context_image] if current_context_image else None
                    # This waits for the Diffuser, but VLM is idle (or we just finished waiting for it)
                    new_b64 = await self.client.call_image_gen(
                        sub_prompt,
                        imgs_payload,
                        seed=candidate_seed
                    )
                
                if not new_b64:
                    continue

                # --- D. START NEXT VERIFICATION ---
                # If we have facts, kick off verification in BACKGROUND
                if visual_facts:
                    pending_candidate_b64 = new_b64 # Keep ref to image
                    # Create task = Fire and Forget (until next loop)
                    verification_task = asyncio.create_task(
                        self.client.verify_image_faithfulness(base64_image=new_b64, facts=visual_facts)
                    )
                else:
                    # If no verification needed (no facts), just accept this image
                    best_candidate_b64 = new_b64
                    break

            # Step B.3: Finalize Winner
            if best_candidate_b64:
                # Log the reasoning for the winner
                t_logger.log_trace(index, "verification_winner", {
                    "score": best_score, 
                    "details": best_verification_details
                })

                # Save the winner to disk
                processed_b64 = await asyncio.to_thread(process_generated_image, best_candidate_b64, save_file)
                
                if processed_b64:
                    state['current_image_b64'] = processed_b64
                    state['prev_image_path'] = str(save_file)
                    # Important: Update local context so subsequent sub-prompts (splits) use this image
                    current_context_image = processed_b64 
                    updates = True
            else:
                 t_logger.warning(f"[RENDER] Index {index}: Failed to generate any valid candidates.")

        # --- 5. FINALIZE ROW ---
        if state['prev_image_path']:
             await dm.update_cell(index, 'img_path', state['prev_image_path'])
             updates = True

        return updates
