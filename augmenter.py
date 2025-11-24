# augmenter.py

import pandas as pd
import numpy as np
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
import os
import sys
import re
import random
import shutil

from src.utils import (
    pil_to_base64,
    base64_to_pil,
    clean_image_artifacts,
    check_server,
    load_config,
    add_padding_to_image
)
from src.api_clients import (
    call_prompt_polisher,
    call_image_gen
)

# Global lock for file writing to prevent CSV corruption
file_write_lock = asyncio.Lock()

# Semaphore to prevent overwhelming the VLM API
VLM_CONCURRENCY = asyncio.Semaphore(10)

# --- MOCKING UTILITIES ---

def generate_random_chaos_prompt(utterance):
    """
    Randomly selects a complex scenario to test the pipeline's 
    handling of splits, zooms, and skips.
    """
    val = random.random()
    
    # 10% Chance: No Change
    if val < 0.1:
        return "[NO_CHANGE]"
    
    # 20% Chance: Zoom Out Sequence
    # This tests: Generation -> Local CPU Zoom -> Generation
    elif val < 0.3:
        return f"Close up shot of {utterance} $$$ [ZOOM_OUT] $$$ Wide angle shot of {utterance}"
    
    # 20% Chance: Multi-Step (No Zoom)
    # This tests: Generation -> Generation (Sequential context)
    elif val < 0.5:
        return f"First angle of {utterance} $$$ Second angle of {utterance}"
    
    # 50% Chance: Standard Single Prompt
    else:
        return f"[Mock Prompt] {utterance}"

async def mock_prompt_polisher(utterance, context=None, previous_prompts=None, images=None):
    """Simulates the VLM returning a prompt."""
    await asyncio.sleep(0.01) # Fast simulation
    
    # If images are present, we are in Stage 2 (Refinement).
    # We usually want to keep the prompt stable here to not break the chaos logic we set in Stage 1.
    if images:
        return f"[Refined] {utterance}"
    
    # If no images, we are in Stage 1 (Initial Creation).
    # Trigger the chaos generator.
    return generate_random_chaos_prompt(utterance)

async def mock_image_gen(prompt, history=None):
    """Simulates Image Gen returning a valid Base64 image."""
    await asyncio.sleep(0.05) 
    
    # Create a 128x128 random colored image
    # We use random colors so you can visually verify that multi-step images change.
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new('RGB', (128, 128), color=color)
    
    d = ImageDraw.Draw(img)
    
    # Draw simple text or shapes to indicate content
    d.rectangle([10, 10, 40, 40], fill="white")
    if "[ZOOM_OUT]" in str(prompt):
        d.text((10, 50), "ZOOM", fill="white")
    
    return pil_to_base64(img)

# -------------------------

def is_no_change_token(s: str) -> bool:
    if s is None:
        return False
    normalized = str(s).strip().upper()
    return normalized in ("[NO_CHANGE]", "NO_CHANGE", "[NO CHANGE]")

def is_zoom_out_token(s: str) -> bool:
    """
    Return True if `s` is a ZOOM_OUT marker.
    """
    if s is None:
        return False
    normalized = str(s).strip().upper()
    return normalized in ("[ZOOM_OUT]", "ZOOM_OUT", "[ZOOM OUT]")

def canonical_no_change() -> str:
    return "[NO_CHANGE]"

def check_servers(args):
    if args.mock:
        print("\n[INFO] Running in MOCK MODE. Skipping server checks.")
        return

    server_config = load_config(config_path='config/server_config.yaml')
    failed_services = []
    polisher_conf = server_config["vlm_service"]
    if not check_server(polisher_conf["gateway"]["host"], polisher_conf["gateway"]["port"]):
        failed_services.append(f"Prompt Polisher service")
    if args.gen_images_from_aug:
        img_gen_conf = server_config["image_gen_service"]
        if not check_server(img_gen_conf["host"], img_gen_conf["port"]):
            failed_services.append(f"Image Gen service")
    if failed_services:
        print(f"[Error] Services down: {failed_services}")
        sys.exit(1)
    print("\nAll required services are running.")

# --- ADAPTER LOGIC ---

def adapt_df_to_standard_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the DataFrame.
    1. Maps 'msg' -> 'text' and 'user' -> 'character'.
    2. Detects 'X_inst' columns and generates 'chunk_X' columns.
    3. Generates 'ctx_start_idx_X' columns (row index where context begins).
    """
    
    # Ensure the DataFrame has a clean RangeIndex for accurate row referencing
    df = df.reset_index(drop=True)

    # --- 1. Column Mapping ---
    if 'msg' in df.columns and 'text' not in df.columns:
        print("  [Adapter] Renaming 'msg' -> 'text'")
        df = df.rename(columns={'msg': 'text'})
    
    if 'user' in df.columns and 'character' not in df.columns:
        print("  [Adapter] Renaming 'user' -> 'character'")
        df = df.rename(columns={'user': 'character'})

    # --- 2. Dynamic Chunk Creation ---
    inst_columns = [c for c in df.columns if c.endswith('_inst')]
    chunk_cols_map = {} 
    
    if inst_columns:
        print(f"  [Adapter] Detected Multi-View format. Found location columns: {inst_columns}")
        
        # PASS 1: Create Chunk IDs
        for col in inst_columns:
            prefix = col.replace('_inst', '')
            chunk_col_name = f"chunk_{prefix}"
            chunk_cols_map[prefix] = chunk_col_name
            
            # Logic: If location changes, increment chunk ID
            loc_series = df[col].fillna("UNKNOWN_LOC")
            condition = loc_series != loc_series.shift()
            
            # Create the chunk ID column
            df[chunk_col_name] = condition.cumsum().fillna(0).astype(int).apply(lambda x: f"{x:03d}")
            print(f"    -> Created '{chunk_col_name}'")

        # PASS 2: Calculate Context Start Indices
        prefixes = list(chunk_cols_map.keys())
        
        if len(prefixes) >= 2:
            print("  [Adapter] Generating Context Start Index columns...")
            
            # Pre-calculate "Start Index" maps for all prefixes
            # This helper series answers: "For any row 'i', at what row index did the CURRENT chunk start?"
            start_index_helpers = {}
            
            for prefix in prefixes:
                col_chunk = chunk_cols_map[prefix]
                
                # 1. Identify where the chunk changed
                mask_changed = df[col_chunk] != df[col_chunk].shift()
                
                # 2. Create a series that holds the *Index Number* only at change points
                helper = pd.Series(np.nan, index=df.index)
                helper[mask_changed] = df.index[mask_changed]
                
                # 3. Forward fill. If row 50 is the same chunk as row 45 (which started at 45),
                # row 50 will now hold the value '45'.
                start_index_helpers[prefix] = helper.ffill().fillna(0).astype(int)

            # Apply Logic to generate the final columns
            for i, prefix_A in enumerate(prefixes):
                # Identify the "Other"
                prefix_B = prefixes[(i + 1) % len(prefixes)]
                
                col_chunk_A = chunk_cols_map[prefix_A]
                col_chunk_B = chunk_cols_map[prefix_B]
                
                # EXPLICIT NAMING: context_start_idx_B
                # Meaning: "When looking at A, here is the index where B's context started"
                target_col_name = f"ctx_start_idx_{prefix_A}"
                
                print(f"    -> Calculating '{target_col_name}' based on shifts in '{col_chunk_A}'")

                # 1. Identify where A shifts
                mask_A_changed = df[col_chunk_A] != df[col_chunk_A].shift()
                
                # 2. Initialize with NaN
                df[target_col_name] = np.nan
                
                # 3. Apply Logic:
                # When A changes, we don't want B's chunk ID. 
                # We want B's START INDEX (which we pre-calculated in start_index_helpers).
                helper_B = start_index_helpers[prefix_B]
                df.loc[mask_A_changed, target_col_name] = helper_B.loc[mask_A_changed]
                
                # 4. Forward Fill (Else condition)
                # If A didn't change, keep the previous reference index.
                df[target_col_name] = df[target_col_name].ffill()
                
                # 5. Handle Index 0 edge case
                if not df.empty:
                    df.at[0, target_col_name] = helper_B.at[0]
                    df[target_col_name] = df[target_col_name].ffill()
                    
                # Optional: Convert to integer for cleaner look (pandas uses float for NaNs by default)
                df[target_col_name] = df[target_col_name].astype(int)

    else:
        if 'chunk_id' not in df.columns:
             print("  [Adapter] No 'chunk_id' or '*_inst' columns found. Defaulting to single chunk.")
             df['chunk_id'] = "001"

    return df

# --- STAGE 1: PARALLEL CHUNK PROCESSING ---

async def process_chunk_create(chunk_id, chunk_df, df, user_perspective, csv_output_path, realistic_chunk=False):
    async with VLM_CONCURRENCY:
        mode_label = "Realistic" if realistic_chunk else "Full"
        print(f"  [Start] Chunk {chunk_id} (Create - {mode_label} Context)")

        previous_prompts_history = []
        chunk_sorted = chunk_df.sort_index()
        user_utterances = chunk_sorted[chunk_sorted['character'] == user_perspective]
        
        # 1. Prepare a global formatted series for slicing based on absolute index
        # We use 'df' (the full dataframe) because the 'start_idx' might theoretically 
        # refer to a row slightly before this chunk if the logic dictates it.
        full_formatted_series = df[['character', 'text']].agg(': '.join, axis=1)

        # 2. Identify the correct context start column (e.g., 'ctx_start_idx_A')
        context_start_col = f"ctx_start_idx_{user_perspective}"
        
        # Safety check: ensure the column exists, otherwise fallback to chunk start
        has_context_idx = context_start_col in df.columns
        if not has_context_idx:
            print(f"    [Warning] Column '{context_start_col}' not found. Falling back to simple chunk slicing.")

        params_changed = False

        for index, row in user_utterances.iterrows():
            current_val = df.at[index, 'initial_prompt']
            
            # --- Existing Skip Logic ---
            if pd.notna(current_val) and str(current_val).strip() != "":
                 if not is_no_change_token(current_val):
                     previous_prompts_history.append(str(current_val).strip())
                 else:
                     df.at[index, 'initial_prompt'] = canonical_no_change()
                 continue

            utterance_text = f"{row['character']}: {row['text']}"

            # --- NEW CONTEXT LOGIC ---
            if has_context_idx:
                # Get the absolute index where this character's context technically began
                start_idx = int(row[context_start_col])
                
                # Define End Point
                # If realistic: strictly history (up to current index)
                # If not realistic: could include future? Usually context is strictly past.
                # We assume 'index' (exclusive) is the cutoff for context.
                end_idx = index 
                
                # Validate indices (prevent negative slicing or look-ahead errors)
                start_idx = max(0, start_idx)
                
                # Slice the full series using the calculated range
                # .loc includes the endpoint, .iloc excludes it. 
                # Since our indices are labels (RangeIndex), .loc is safer but includes the end.
                # We want start_idx (inclusive) to index (exclusive).
                # Let's use boolean masking on the index for clarity.
                mask = (full_formatted_series.index >= start_idx) & (full_formatted_series.index < end_idx)
                current_context = full_formatted_series.loc[mask].tolist()
            
            else:
                # Fallback to original logic (Chunk-based) if column missing
                if realistic_chunk:
                    # Formatted dialogue *within this chunk* up to now
                    chunk_formatted = chunk_sorted[['character', 'text']].agg(': '.join, axis=1)
                    current_context = chunk_formatted[chunk_formatted.index < index].tolist()
                else:
                    # Whole chunk
                    chunk_formatted = chunk_sorted[['character', 'text']].agg(': '.join, axis=1)
                    current_context = chunk_formatted.tolist()

            # Debugging context boundaries (Optional)
            # print(f"    [Ctx Debug] Index {index-3} | Start: {start_idx} | Len: {index - start_idx - 1}") # NOT WORKING RN TO IMPROVE
            print(f"    [Ctx Debug] Utterance: {utterance_text} | Current Context: {current_context}")


            # --- Call Polisher ---
            try:
                polished_prompt = await call_prompt_polisher(
                    utterance=utterance_text,
                    context=current_context,
                    previous_prompts=previous_prompts_history if previous_prompts_history else None
                )

                if polished_prompt:
                    if is_no_change_token(polished_prompt):
                        polished_prompt = canonical_no_change()

                    df.at[index, 'initial_prompt'] = polished_prompt
                    params_changed = True

                    if not is_no_change_token(polished_prompt):
                        previous_prompts_history.append(polished_prompt)
                else:
                    df.at[index, 'initial_prompt'] = ""

            except Exception as e:
                print(f"    Error Chunk {chunk_id} Index {index}: {e}")

        if params_changed:
            async with file_write_lock:
                df.to_csv(csv_output_path, index=False)

        print(f"  [Done] Chunk {chunk_id} (Create)")


async def create_aug(df: pd.DataFrame, user_perspective: str, csv_output_path: str, realistic_chunk: bool, chunk_col: str) -> pd.DataFrame:
    mode_label = "Realistic" if realistic_chunk else "Full"
    print(f"--- Stage 1: Parallel Contextual Prompts ({user_perspective}) [Grouping: {chunk_col}] ---")
    if 'initial_prompt' not in df.columns:
        df['initial_prompt'] = pd.NA

    # DYNAMIC GROUPING
    grouped = df.groupby(chunk_col)
    tasks = []

    for chunk_id, chunk_df in grouped:
        tasks.append(
            process_chunk_create(chunk_id, chunk_df, df, user_perspective, csv_output_path, realistic_chunk)
        )

    await asyncio.gather(*tasks)
    return df


# --- STAGE 2: PARALLEL RENDERING (WITH MULTI-STEP LOGIC) ---

async def process_chunk_render(chunk_id, chunk_df, df, user_perspective, output_dir_path, csv_output_path):
    """
    Processes a single chunk for Stage 2.
    Runs fully parallel. The Image Gen server now queues requests internally.
    """
    print(f"  [Start] Chunk {chunk_id} (Render)")

    current_image_b64 = None
    chunk_sorted = chunk_df.sort_index()
    params_changed = False

    for index, row in chunk_sorted.iterrows():
        if (row['character'] != user_perspective) or pd.isna(row['initial_prompt']):
            continue

        if pd.notna(row['img_path']) and str(row['img_path']).strip() != "":
            if os.path.exists(row['img_path']):
                try:
                    with Image.open(row['img_path']) as img:
                        current_image_b64 = pil_to_base64(img)
                    continue
                except:
                    pass

        initial_prompt = str(row['initial_prompt']).strip()

        if is_no_change_token(initial_prompt):
            df.at[index, 'initial_prompt'] = canonical_no_change()
            continue

        # 1. Visual Refinement
        final_prompt = initial_prompt
        if current_image_b64:
            async with VLM_CONCURRENCY:
                try:
                    refined = await call_prompt_polisher(initial_prompt, images=[current_image_b64])
                    if refined:
                        if is_no_change_token(refined):
                            df.at[index, 'final_prompt'] = canonical_no_change()
                            continue
                        final_prompt = refined
                except Exception as e:
                    print(f"    Refine Error {index}: {e}")

        df.at[index, 'final_prompt'] = final_prompt

        # 2. Multi-Step Processing
        sub_prompts = [p.strip() for p in final_prompt.split("$$$") if p.strip()]

        if not sub_prompts:
            print(f"    Warning: Empty prompts at index {index}")
            continue

        last_saved_path = None

        for step_i, sub_prompt in enumerate(sub_prompts):
            suffix = f"_seq{step_i}" if len(sub_prompts) > 1 else ""
            img_filename = f"chunk_{chunk_id}_{user_perspective}_{index}{suffix}.png"
            save_path = os.path.join(output_dir_path, img_filename)

            # --- ZOOM OUT LOGIC ---
            if is_zoom_out_token(sub_prompt):
                if current_image_b64:
                    try:
                        current_pil = base64_to_pil(current_image_b64)
                        zoomed_pil = add_padding_to_image(current_pil, scale_factor=0.8, fill_color="white")
                        
                        if zoomed_pil:
                            zoomed_pil.save(save_path)
                            current_image_b64 = pil_to_base64(zoomed_pil)
                            last_saved_path = str(save_path)
                            params_changed = True
                    except Exception as e:
                        print(f"    Zoom Error index {index}: {e}")
                continue

            # --- STANDARD GENERATION (OR MOCK) ---
            try:
                new_image_b64 = await call_image_gen(
                    sub_prompt,
                    [current_image_b64] if current_image_b64 else None
                )

                if new_image_b64:
                    img_pil = clean_image_artifacts(base64_to_pil(new_image_b64))
                    img_pil.save(save_path)

                    current_image_b64 = pil_to_base64(img_pil)
                    last_saved_path = str(save_path)
                    params_changed = True
            except Exception as e:
                 print(f"    Render Error {index} step {step_i}: {e}")
                 break

        if last_saved_path:
            df.at[index, 'img_path'] = last_saved_path

        if params_changed:
            async with file_write_lock:
                 df.to_csv(csv_output_path, index=False)

    print(f"  [Done] Chunk {chunk_id} (Render)")


async def gen_images_from_aug(df: pd.DataFrame, user_perspective: str, images_output_dir: str, csv_output_path: str, chunk_col: str) -> pd.DataFrame:
    print(f"--- Stage 2: Parallel Rendering ({user_perspective}) [Grouping: {chunk_col}] ---")
    output_dir_path = Path(images_output_dir) / user_perspective
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if 'final_prompt' not in df.columns: df['final_prompt'] = pd.NA
    if 'img_path' not in df.columns: df['img_path'] = pd.NA
    if 'index' not in df.columns: df['index'] = df.index

    # DYNAMIC GROUPING
    grouped = df.groupby(chunk_col)
    tasks = []

    for chunk_id, chunk_df in grouped:
        tasks.append(
            process_chunk_render(chunk_id, chunk_df, df, user_perspective, str(output_dir_path), csv_output_path)
        )

    await asyncio.gather(*tasks)
    return df


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--user", type=str)
    parser.add_argument("--automatic_users", action='store_true')
    parser.add_argument("--create_aug", action='store_true')
    parser.add_argument("--gen_images_from_aug", action='store_true')
    parser.add_argument("--aug_output_path", type=str, default="output/")
    parser.add_argument("--images_output_path", type=str, default="output/")
    parser.add_argument("--realistic_chunk", action='store_true', 
                        help="If set, context provided to VLM is strictly previous utterances.")
    parser.add_argument("--mock", action='store_true',
                        help="Run in mock mode (no API calls, generates chaos data).")
    
    args = parser.parse_args()

    if not args.file:
        print("Error: --file argument is required.")
        sys.exit(1)

    if not Path(args.file).exists():
        print(f"[Error] Input file not found: {args.file}")
        sys.exit(1)

    # --- MOCK SETUP ---
    if args.mock:
        global call_prompt_polisher, call_image_gen
        call_prompt_polisher = mock_prompt_polisher
        call_image_gen = mock_image_gen
        print(">>> MOCK MODE ENGAGED: CHAOS PROMPTS ACTIVE <<<")
            
        # Derive Paths based on Input Filename
        input_stem = Path(args.file).stem
        mock_root = Path("output/mock")
        args.aug_output_path = mock_root / f"{input_stem}_aug.csv"
        args.images_output_path = mock_root / f"{input_stem}_images"

        print(f"    Mock Output File: {args.aug_output_path}")
        print(f"    Mock Images Dir:  {args.images_output_path}")

        # Clean SPECIFIC artifacts for this file (Fresh Run)
        # We do NOT wipe the whole mock_root, just this file's results
        if args.aug_output_path.exists():
            print("    Cleaning existing mock CSV...")
            args.aug_output_path.unlink()
            
        if args.images_output_path.exists():
            print("    Cleaning existing mock images folder...")
            shutil.rmtree(args.images_output_path)
        
        # Create the root folder if it doesn't exist
        mock_root.mkdir(parents=True, exist_ok=True)

    else:
        if not args.file:
             print("Error: --file argument is required when not in --mock mode.")
             sys.exit(1)
             
        check_servers(args)

        input_path = Path(args.file)
        if args.aug_output_path == "output/":
            args.aug_output_path = Path("output") / f"{input_path.stem}_aug{input_path.suffix}"
        else:
            args.aug_output_path = Path(args.aug_output_path)
        if args.gen_images_from_aug and args.images_output_path == "output/":
            args.images_output_path = Path("output") / f"{input_path.stem}_images"

    # --- LOAD AND ADAPT ---
    if args.aug_output_path.exists():
        print(f"Resuming from {args.aug_output_path}")
        df = pd.read_csv(args.aug_output_path, dtype=str) # Load as string to be safe
    else:
        print(f"Loading Source: {args.file}")
        df = pd.read_csv(args.file)

        # --- SAFETY FIX: Ensure existing chunk_id is padded string ---
        if 'chunk_id' in df.columns:
            # If pandas loaded "001" as int 1, this turns it back to "001"
            df['chunk_id'] = df['chunk_id'].astype(str).str.replace(r'\.0$', '', regex=True) # Handle potential float loading
            df['chunk_id'] = df['chunk_id'].apply(lambda x: x.zfill(3))

        # RUN ADAPTER ONLY ON FIRST LOAD
        df = adapt_df_to_standard_format(df)
        
        # Save immediately to establish the schema with new columns
        args.aug_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.aug_output_path, index=False)

    # Determine Users
    users_to_process = []
    if args.automatic_users:
        users_to_process = [u for u in df['character'].unique() if pd.notna(u)]
    elif args.user:
        users_to_process = [args.user]

    # --- PROCESS PER USER ---
    for user in users_to_process:
        # Determine which chunk column to use for this user
        # 1. Check if specific chunk column exists (e.g. chunk_A for user A)
        chunk_col = f"chunk_{user}"
        
        # 2. If not found, fall back to standard 'chunk_id'
        if chunk_col not in df.columns:
            chunk_col = 'chunk_id'
        
        # 3. Validation
        if chunk_col not in df.columns:
             print(f"Warning: No valid chunk column found for user {user}. Skipping.")
             continue

        if args.create_aug:
            df = await create_aug(df, user, str(args.aug_output_path), args.realistic_chunk, chunk_col)

        if args.gen_images_from_aug:
            df = await gen_images_from_aug(df, user, str(args.images_output_path), str(args.aug_output_path), chunk_col)

if __name__ == "__main__":
    asyncio.run(main())
