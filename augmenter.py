# augmenter.py

import pandas as pd
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image
import os
import sys
import re

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

def is_no_change_token(s: str) -> bool:
    """
    Return True if `s` is a NO_CHANGE marker.
    """
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


# --- STAGE 1: PARALLEL CHUNK PROCESSING ---

async def process_chunk_create(chunk_id, chunk_df, df, user_perspective, csv_output_path, realistic_chunk=False):
    async with VLM_CONCURRENCY:
        mode_label = "Realistic" if realistic_chunk else "Full"
        print(f"  [Start] Chunk {chunk_id} (Create - {mode_label} Context)")

        previous_prompts_history = []
        
        chunk_sorted = chunk_df.sort_index()
        user_utterances = chunk_sorted[chunk_sorted['character'] == user_perspective]
        full_context_list = chunk_sorted['text'].tolist()

        params_changed = False

        for index, row in user_utterances.iterrows():
            # Skip if exists
            current_val = df.at[index, 'initial_prompt']
            if pd.notna(current_val) and str(current_val).strip() != "":
                 if not is_no_change_token(current_val):
                     previous_prompts_history.append(str(current_val).strip())
                 else:
                     df.at[index, 'initial_prompt'] = canonical_no_change()
                 continue

            utterance_text = row['text']

            # Realistic vs Oracle Context
            if realistic_chunk:
                current_context = chunk_sorted[chunk_sorted.index < index]['text'].tolist()
            else:
                current_context = full_context_list

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


async def create_aug(df: pd.DataFrame, user_perspective: str, csv_output_path: str, realistic_chunk: bool) -> pd.DataFrame:
    mode_label = "Realistic" if realistic_chunk else "Full"
    print(f"--- Stage 1: Parallel Contextual Prompts ({user_perspective}) [Context: {mode_label}] ---")
    if 'initial_prompt' not in df.columns:
        df['initial_prompt'] = pd.NA

    grouped = df.groupby('chunk_id')
    tasks = []

    for chunk_id, chunk_df in grouped:
        tasks.append(
            process_chunk_create(chunk_id, chunk_df, df, user_perspective, csv_output_path, realistic_chunk)
        )

    await asyncio.gather(*tasks)
    return df


# --- STAGE 2: PARALLEL RENDERING ---

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

        img_filename = f"chunk_{chunk_id}_{user_perspective}_{index}.png"
        save_path = os.path.join(output_dir_path, img_filename)

        # Check existing image to resume context
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
            # Standardize formatting in the DF
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

        # 2. Image Generation
        try:
            new_image_b64 = await call_image_gen(
                final_prompt,
                [current_image_b64] if current_image_b64 else None
            )

            if new_image_b64:
                img_pil = clean_image_artifacts(base64_to_pil(new_image_b64))
                img_pil.save(save_path)

                df.at[index, 'img_path'] = str(save_path)
                current_image_b64 = pil_to_base64(img_pil)
                params_changed = True
            else:
                print(f"    Render None {index}")
        except Exception as e:
             print(f"    Render Error {index}: {e}")

        if params_changed:
            async with file_write_lock:
                 df.to_csv(csv_output_path, index=False)

    print(f"  [Done] Chunk {chunk_id} (Render)")


async def gen_images_from_aug(df: pd.DataFrame, user_perspective: str, images_output_dir: str, csv_output_path: str) -> pd.DataFrame:
    print(f"--- Stage 2: Parallel Rendering ({user_perspective}) ---")

    output_dir_path = Path(images_output_dir) / user_perspective
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if 'final_prompt' not in df.columns:
        df['final_prompt'] = pd.NA
    if 'img_path' not in df.columns:
        df['img_path'] = pd.NA
    if 'index' not in df.columns:
        df['index'] = df.index

    grouped = df.groupby('chunk_id')
    tasks = []

    for chunk_id, chunk_df in grouped:
        tasks.append(
            process_chunk_render(chunk_id, chunk_df, df, user_perspective, str(output_dir_path), csv_output_path)
        )

    await asyncio.gather(*tasks)
    return df


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--user", type=str)
    parser.add_argument("--automatic_users", action='store_true')
    parser.add_argument("--create_aug", action='store_true')
    parser.add_argument("--gen_images_from_aug", action='store_true')
    parser.add_argument("--aug_output_path", type=str, default="output/")
    parser.add_argument("--images_output_path", type=str, default="output/")
    parser.add_argument("--realistic_chunk", action='store_true', 
                        help="If set, context provided to VLM is strictly previous utterances.")
    
    args = parser.parse_args()

    check_servers(args)

    input_path = Path(args.file)
    if args.aug_output_path == "output/":
        args.aug_output_path = Path("output") / f"{input_path.stem}_aug{input_path.suffix}"
    else:
        args.aug_output_path = Path(args.aug_output_path)

    if args.gen_images_from_aug and args.images_output_path == "output/":
        args.images_output_path = Path("output") / f"{input_path.stem}_images"

    # Smart Load
    if args.aug_output_path.exists():
        print(f"Resuming from {args.aug_output_path}")
        df = pd.read_csv(args.aug_output_path, dtype={'chunk_id': str})
    else:
        print(f"Loading {args.file}")
        df = pd.read_csv(args.file, dtype={'chunk_id': str})

    users_to_process = []
    if args.automatic_users:
        users_to_process = [u for u in df['character'].unique() if pd.notna(u)]
    elif args.user:
        users_to_process = [args.user]

    for user in users_to_process:
        if args.create_aug:
            df = await create_aug(df, user, str(args.aug_output_path), args.realistic_chunk)

        if args.gen_images_from_aug:
            df = await gen_images_from_aug(df, user, str(args.images_output_path), str(args.aug_output_path))


if __name__ == "__main__":
    asyncio.run(main())
