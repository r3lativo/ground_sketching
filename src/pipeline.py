# src/pipeline.py

import asyncio
import random
from pathlib import Path

from src.utils import (
    load_existing_image,
    process_zoom,
    process_generated_image,
    is_not_empty_val
)
from src.api_clients import Action


class AugmentationPipeline:
    def __init__(self, api_client):
        self.client = api_client

    async def run_single_user(
        self,
        data_manager,
        user: str,
        vlm_semaphore: asyncio.Semaphore,
        img_semaphore: asyncio.Semaphore,
        t_logger,
        pipeline_config: dict,
        pbar=None
    ):
        """
        Runs the pipeline for ONE specific user in a specific file.
        """
        t_logger.info(f"--- Starting Pipeline for User: {user} ---")
        
        # Unpack Config
        create = pipeline_config.get('create')
        render = pipeline_config.get('render')
        oracle = pipeline_config.get('oracle')
        relation_triplets = pipeline_config.get('relation_triplets')
        candidate_count = pipeline_config.get('candidate_count')
        
        img_out_dir = pipeline_config.get('img_output_dir')

        # Setup Render Path
        user_out_path = None
        if render and img_out_dir:
            user_out_path = Path(img_out_dir) / user
            user_out_path.mkdir(parents=True, exist_ok=True)

        sorted_indices = sorted(data_manager.df.index.tolist())
        
        # --- STATE TRACKING ---
        # frame_idx: No longer manually tracked for ID generation, but kept for logging/filename logic if needed.
        # seq_idx: Increments on [CONTINUE] or sub-prompts ($$$).
        state = {
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

        # --- A. CREATE AND RENDER ---
        for index in sorted_indices:
            try:
                row = data_manager.df.loc[index]
            except KeyError:
                continue
            
            if row['character'] != user:
                continue

            # PHASE 1: CREATE
            if create:
                await self._phase_create(
                    data_manager, index, user, oracle,
                    vlm_semaphore, t_logger
                )
                if pbar: pbar.update(1)

            # PHASE 2: RENDER (With Verification)
            if render and user_out_path:
                await self._phase_render(
                    data_manager, index, user, oracle, 
                    state, user_out_path, 
                    vlm_semaphore, img_semaphore, t_logger,
                    candidate_count
                )
                if pbar: pbar.update(1)
            
            # Save periodically
            await data_manager.save()

        # --- B. TRIPLET RELATION ---
        for index in sorted_indices:
            try: row = data_manager.df.loc[index]
            except KeyError: continue
            
            if row['character'] != user: continue

            if relation_triplets:
                await self._phase_relations(
                    data_manager, index, user,
                    vlm_semaphore, t_logger
                )
                if pbar: pbar.update(1)

            # Save periodically
            await data_manager.save()

        # Final save
        await data_manager.save()
        t_logger.info(f"--- Finished User: {user} ---")

    async def _update_cells(self, dm, index, decision):
        """Helper to update all relevant cells for meta extraction"""
        await dm.update_cell(index, 'frame_choice', decision.action)
        await dm.update_cell(index, 'frame_meta', decision.frame_meta)
        await dm.update_cell(index, 'relation', decision.relation)
        await dm.update_cell(index, 'imagery', decision.imagery)
        await dm.update_cell(index, 'initial_prompt', decision.imagery) # This will be modified in the render phase

    async def _phase_create(self, dm, index, user, oracle, vlm_sem, t_logger):
        """
        Handles Logic: VLM Decision -> Strategy Execution -> Prompt Update -> Frame ID Generation
        """
        row = dm.df.loc[index]

        # --- SKIP LOGIC ---
        frame_val = str(row.get('frame_choice', ''))
        prompt_val = row.get('initial_prompt')
        
        # 1. Check if we already decided to SKIP this frame
        if frame_val == Action.SKIP:
            t_logger.info(f"[CREATE] Index {index}: Found existing [SKIP]. Skipping.")
            return True
            
        # 2. Check if we have a valid prompt for NEW/CONTINUE
        # We only skip if BOTH the frame choice AND the prompt exist
        if dm._is_not_empty_val(frame_val) and dm._is_not_empty_val(prompt_val):
            t_logger.info(f"[CREATE] Index {index}: Found existing Prompt & Frame. Skipping.")
            return True

        utterance = f"{row['character']}: {row['text']}"
        context = dm.get_context_for_index(index, user, oracle, include_prev=True)
        prev_prompts = dm.get_prev_prompts_for_frame(index, user, oracle)

        ### META PHASE ###
        # 1. Get Strategy (Throttled)
        async with vlm_sem:
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
            # Check if *this* user has ever had a [NEW] frame in the past
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
            await self._update_cells(dm, index, decision)
            return False

        # 4. Handle NEW vs KEEP
        if decision.action == Action.NEW:
            prev_prompts = [] # Reset prompt context for the strategy generation
        
        # If there is an imagery, we pass that as the utterance to model
        if decision.imagery:
            utterance = decision.imagery

        # Write Metadata
        await self._update_cells(dm, index, decision)

        # --- IMMEDIATE FRAME ID GENERATION ---
        # If this is a [NEW] frame, we must calculate and assign the ID immediately
        # so subsequent steps (Render/Relations) can find it.
        if decision.action == Action.NEW:
            # We call the DataManager to recalculate IDs for this user up to this point
            dm.ensure_frame_ids(user)
            # Retrieve the newly generated ID for logging purposes
            new_id = dm.df.at[index, 'frame_id']
            t_logger.info(f"[CREATE] Index {index}: Generated Frame ID: {new_id}")

        ### INITIAL PROMPT PHASE ###
        # 5. Execute Strategy
        new_prompt_data = None
        
        async with vlm_sem:
            t_logger.info(f"[CREATE] Index {index}: Executing Strategy {decision.strategy_name}...")
            
            # Pass the schema explicitly
            new_prompt_data = await self.client.execute_vlm_strategy(
                strategy=decision.strategy, 
                utterance=utterance, 
                context=context, 
                previous_prompts=prev_prompts, 
                validation_schema=decision.validation_schema
            )
            t_logger.log_trace(index, "strategy_generation", {"raw_data": new_prompt_data})

            if new_prompt_data is None:
                t_logger.error(f"[CREATE] Index {index}: Strategy failed to generate prompt. Skipping update.")
                return False

        # EXTRACT STRING FROM DICT
        # Validation ensures 'scene' key exists if new_prompt_data is not None
        final_prompt_str = new_prompt_data.get('scene', "")

        await dm.update_cell(index, 'initial_prompt', final_prompt_str)
        return True

    async def _phase_render(self, dm, index, user, oracle, state, out_path, vlm_sem, img_sem, t_logger, candidate_count):
        """
        Handles Logic: State Update -> Refinement -> Best-of-N Generation -> Save
        """
        row = dm.df.loc[index]
        initial_prompt = str(row.get('initial_prompt', '')).strip()
        frame_choice = str(row.get('frame_choice', '')).strip()
        
        # --- 1. HANDLE SKIP ACTION ---
        if Action.SKIP in frame_choice:
             t_logger.info(f"[RENDER] Index {index}: Frame choice is SKIP. Skipping generation.")
             return False

        if not is_not_empty_val(initial_prompt):
            return False
        
        # --- SKIP LOGIC: If we already have a valid image path, skip generation ---
        existing_img_path = row.get('img_path')
        if dm._is_not_empty_val(existing_img_path):
             # Optional: Check if file actually exists on disk
             if Path(str(existing_img_path)).exists():
                 t_logger.info(f"[RENDER] Index {index}: Found existing image at {existing_img_path}. Skipping.")
                 
                 # IMPORTANT: We still need to update the state for the next rows!
                 # Load the image so subsequent [CONTINUE] frames can use it as context.
                 loaded = await asyncio.to_thread(load_existing_image, str(existing_img_path))
                 if loaded:
                     state['current_image_b64'] = loaded
                     state['prev_image_path'] = str(existing_img_path)
                     
                 # Reset sequence logic if NEW
                 if Action.NEW in str(row.get('frame_choice', '')):
                     state['seq_idx'] = 1
                 else:
                     state['seq_idx'] += 1
                     
                 return False

        # --- 2. UPDATE STATE (Frame & Seq) ---
        is_new_frame = (Action.NEW in frame_choice)

        # [NEW LOGIC] Retrieve the actual Frame ID string (e.g., "A_1") from the DataManager
        current_frame_id_str = row.get('frame_id')
        
        # Fallback: If for some reason ID is missing (should be fixed by _phase_create), 
        # try to parse it or log warning.
        if not dm._is_not_empty_val(current_frame_id_str) and is_new_frame:
             t_logger.warning(f"[RENDER] Index {index}: [NEW] frame missing 'frame_id'. Attempting repair.")
             dm.ensure_frame_ids(user)
             current_frame_id_str = dm.df.at[index, 'frame_id']

        # Extract number for filename (optional, depends on your naming convention)
        # Assuming format "USER_NUMBER"
        try:
            frame_num = int(str(current_frame_id_str).split('_')[-1])
        except (ValueError, IndexError):
            # Fallback for logging if parsing fails
            frame_num = "Unknown"

        if is_new_frame:
            state['seq_idx'] = 1
            current_context_image = None
            t_logger.info(f"[RENDER] Index {index}: [NEW] frame detected. ID: {current_frame_id_str}.")
        else:
            current_context_image = state['current_image_b64']
            t_logger.info(f"[RENDER] Index {index}: [CONTINUE] frame detected. ID: {current_frame_id_str}, Seq {state['seq_idx']}.")

        # --- 3. REFINE PROMPT (FIXED LOGIC) ---
        # We must handle multi-step prompts ($$$) BEFORE refinement.
        # Otherwise, the VLM sees a giant string and might destroy the separators.
        
        raw_sub_prompts = [p.strip() for p in initial_prompt.split("$$$") if p.strip()]
        refined_sub_prompts = []
        
        # Check if we should refine (Visual Context exists + Not a New Frame)
        should_refine = (current_context_image is not None) and (not is_new_frame)
        
        if should_refine:
            async with vlm_sem:
                mm_decision = await self.client.get_meta_and_strategy(is_oracle=oracle, has_images=True)
                
                if mm_decision.strategy:
                    t_logger.info(f"[RENDER] Index {index}: Refining {len(raw_sub_prompts)} sub-prompts with visual context...")
                    
                    for i, sub_p in enumerate(raw_sub_prompts):
                        # Special handling: Don't refine structural commands like [ZOOM_OUT]
                        if "[ZOOM_OUT]" in sub_p.upper():
                            refined_sub_prompts.append(sub_p)
                            continue

                        # Refine the specific segment
                        refined_segment = await self.client.execute_vlm_strategy(
                            mm_decision.strategy, sub_p, images=[current_context_image]
                        )
                        
                        if refined_segment:
                            refined_sub_prompts.append(refined_segment)
                            t_logger.log_trace(index, f"refinement_step_{i+1}", {"before": sub_p, "after": refined_segment})
                        else:
                            # Fallback if VLM fails
                            refined_sub_prompts.append(sub_p)
                else:
                    # Strategy lookup failed, keep original
                    refined_sub_prompts = raw_sub_prompts
        else:
            # No refinement needed
            refined_sub_prompts = raw_sub_prompts

        # Reconstruct the final prompt string for the CSV
        final_prompt = " $$$ ".join(refined_sub_prompts)
        await dm.update_cell(index, 'final_prompt', final_prompt)

        # --- 4. GENERATE & VERIFY LOOP ---
        # Now we use the list we just built, preserving the order and separation
        sub_prompts = refined_sub_prompts
        updates = False

        for sub_prompt in sub_prompts:
            # [NEW LOGIC] Use the actual Frame ID string in the filename
            # Filename: User_FrameIdNumber_SeqX.png (e.g., A_1_seq1.png)
            img_filename = f"{user}_{frame_num}_seq{state['seq_idx']}.png"
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
            async with vlm_sem:
                 visual_facts = await self.client.call_prompt_summarizer(full_history)
            
            # Step B.2: Loop
            best_candidate_b64 = None
            best_score = -1.0
            best_verification_details = []
            
            # Queue to hold the running verification task
            verification_task = None
            pending_candidate_b64 = None # The image currently being verified

            # We loop one extra time to allow the final verification to finish
            actual_loops = candidate_count if visual_facts else 1
            
            for i in range(actual_loops + 1):
                
                # --- PARALLEL EXECUTION DEFINITION ---
                # We define the generation step as a coroutine here.
                # It will run concurrently with the verification of the PREVIOUS image.
                async def _gen_step():
                    if i >= actual_loops: return None
                    t_logger.info(f"[RENDER] Generating Candidate {i+1}/{actual_loops}...")
                    candidate_seed = random.randint(0, 2**32 - 1)
                    async with img_sem:
                        imgs_payload = [current_context_image] if current_context_image else None
                        return await self.client.call_image_gen(
                            sub_prompt, imgs_payload, seed=candidate_seed
                        )

                # --- START CONCURRENT TASKS ---
                # Task List: [0] Generation, [1] Verification (Optional)
                current_tasks = [_gen_step()]
                if verification_task:
                    current_tasks.append(verification_task)

                # Wait for BOTH the next image to generate AND the previous one to verify
                results = await asyncio.gather(*current_tasks)

                # --- PROCESS RESULTS ---
                
                # 1. Result of Generation (Always index 0)
                new_b64 = results[0]

                # 2. Result of Verification (Index 1, if it existed)
                if verification_task:
                    score, details = results[1]
                    t_logger.info(f"[RENDER] Candidate Score: {score:.2f}")

                    if score > best_score:
                        best_score = score
                        best_candidate_b64 = pending_candidate_b64
                        best_verification_details = details
                    
                    # Early Stopping: If perfect, we stop immediately
                    if score >= 1.0:
                        t_logger.info(f"[RENDER] Perfect score achieved. Stopping early.")
                        break

                # --- PREPARE FOR NEXT LOOP ---
                if new_b64 is None: break

                if visual_facts:
                    # If we have facts, we queue the NEW image for verification in the NEXT loop
                    pending_candidate_b64 = new_b64
                    verification_task = asyncio.create_task(
                        self.client.verify_image_faithfulness(base64_image=new_b64, facts=visual_facts)
                    )
                else:
                    # If no facts exist (no verification needed), this image is automatically the winner
                    best_candidate_b64 = new_b64
                    break

            # Step B.3: Finalize Winner
            if best_candidate_b64:
                t_logger.log_trace(index, "verification_winner", {
                    "score": best_score, 
                    "details": best_verification_details
                })

                # Save the winner to disk
                processed_b64 = await asyncio.to_thread(process_generated_image, best_candidate_b64, save_file)
                
                if processed_b64:
                    state['current_image_b64'] = processed_b64
                    state['prev_image_path'] = str(save_file)
                    current_context_image = processed_b64 
                    updates = True
            else:
                 t_logger.warning(f"[RENDER] Index {index}: Failed to generate any valid candidates.")

        # --- 5. FINALIZE ROW ---
        if state['prev_image_path']:
             await dm.update_cell(index, 'img_path', state['prev_image_path'])
             updates = True

        return updates
    
    async def _phase_relations(self, dm, index, user, vlm_sem, t_logger):
        row = dm.df.loc[index]
        relation_raw = row.get('relation')

        # --- SKIP LOGIC ---
        if dm._is_not_empty_val(row.get('extracted_triplets')):
            t_logger.info(f"[RELATIONS] Index {index}: Triplets already extracted. Skipping.")
            return False

        # 0. Skip if no relation is defined
        if not dm._is_not_empty_val(relation_raw):
            return False

        # 1. [NEW LOGIC] Frame IDs are already ensured in _phase_create.
        # However, for safety (e.g. running relations on an old CSV), we can check quickly.
        if not dm._is_not_empty_val(row.get('frame_id')):
             dm.ensure_frame_ids(user)

        # 2. Retrieve Neighborhood
        neighborhood = dm.get_frame_neighborhood(index, user)
        
        # [Edge Case] Isolated Frame Check
        # Logic: If NO previous history AND NO future context, 
        #        cannot form a triplet relation to anything.
        p_txt = neighborhood.get('prev_text')
        n_txt = neighborhood.get('next_text')
        
        if not p_txt and not n_txt:
             t_logger.info(f"[RELATIONS] Index {index}: Isolated frame (No Prev/Next Text). Skipping extraction.")
             return False

        t_logger.info(f"[RELATIONS] Index {index}: Processing relation '{relation_raw}'")

        # 3. Prepare Prompt
        text_combined = f"{row['character']}: {row['text']}"
        extracted_data = None
        
        # 4. Execute
        async with vlm_sem:
            cid = neighborhood['current_frame_id']
            t_logger.info(f"[RELATIONS] Extracting triplets for Frame {cid}...")
            
            extracted_data = await self.client.call_triplets_extraction(
                utterance=relation_raw,
                context=neighborhood
            )

        # 5. Save with Fallback
        if extracted_data:
            clean_triplets = []
            
            # A. Attempt Strict Cleaning
            if isinstance(extracted_data, list):
                for item in extracted_data:
                    if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                        clean_triplets.append((item['subject'], item['predicate'], item['object']))
            
            # B. Decide what to save
            if clean_triplets:
                final_value = str(clean_triplets)
                t_logger.info(f"[RELATIONS] Extracted {len(clean_triplets)} triplets.")
            else:
                final_value = str(extracted_data)
                t_logger.warning(f"[RELATIONS] Extraction format invalid. Saving raw output: {final_value[:50]}...")

            await dm.update_cell(index, 'extracted_triplets', final_value)
            return True

        return False
