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

from src.utils import (
    pil_to_base64,
    base64_to_pil,
    clean_image_artifacts,
    check_server,
    load_config
)
from src.api_clients import (
    call_prompt_polisher,
    call_image_gen
)

def check_servers(args):
    """
    Checks if all required backend services are running.
    """
    server_config = load_config(config_path='config/server_config.yaml')
    failed_services = []

    # --- Check Prompt Polisher ---
    polisher_conf = server_config["vlm_service"]
    if not check_server(polisher_conf["gateway"]["host"], polisher_conf["gateway"]["port"]):
        failed_services.append(
            f"Prompt Polisher service at http://{polisher_conf['gateway']['host']}:{polisher_conf['gateway']['port']}"
        )

    # --- Check Image Gen ---
    if args.gen_images_from_aug:
        img_gen_conf = server_config["image_gen_service"]
        if not check_server(img_gen_conf["host"], img_gen_conf["port"]):
            failed_services.append(
                f"Image Gen service at http://{img_gen_conf['host']}:{img_gen_conf['port']}"
            )

    if failed_services:
        print("\n[Error] One or more required services are down:")
        for service_msg in failed_services:
            print(f"- {service_msg}")
        print("Exiting.")
        sys.exit(1)
    
    print("\nAll required services are running.")


async def create_aug(df: pd.DataFrame, user_perspective: str, csv_output_path: str) -> pd.DataFrame:
    """
    Stage 1: Text Context -> Initial Image Prompt
    Saves to CSV after every row.
    """
    print(f"--- Stage 1: Creating contextual prompts for user: {user_perspective} ---")

    # Initialize column if not present
    if 'initial_prompt' not in df.columns:
        df['initial_prompt'] = pd.NA

    grouped = df.groupby('chunk_id')

    for chunk_id, chunk_df in grouped:
        print(f"Processing Chunk {chunk_id} (Text -> Prompt)")
        
        previous_prompts_history = []
        
        # Get utterances for the specific user
        user_utterances = chunk_df[
            chunk_df['character'] == user_perspective
        ].sort_index()

        context_list = chunk_df.sort_index()['text'].tolist()
        
        for index, row in user_utterances.iterrows():
            utterance_text = row['text']
            
            try:
                polished_prompt = await call_prompt_polisher(
                    utterance=utterance_text,
                    context=context_list,
                    previous_prompts=previous_prompts_history if previous_prompts_history else None
                )

                if polished_prompt:
                    # Update DataFrame
                    df.at[index, 'initial_prompt'] = polished_prompt
                    
                    if polished_prompt != "[NO_CHANGE]":
                        print(f"  > [{index}] Prompt created.")
                        previous_prompts_history.append(polished_prompt)
                    else:
                        print(f"  > [{index}] [NO_CHANGE]")
                else:
                    print(f"  > [{index}] Failed to generate prompt.")
                
                # --- INCREMENTAL SAVE ---
                # Save after every attempt so we don't lose progress
                df.to_csv(csv_output_path, index=False)

            except Exception as e:
                print(f"Error index {index}: {e}")
                
    return df


async def gen_images_from_aug(
    df: pd.DataFrame, 
    user_perspective: str, 
    images_output_dir: str,
    csv_output_path: str
) -> pd.DataFrame:
    """
    Stage 2: (Initial Prompt + Previous Image) -> Final Prompt -> Image
    Saves to CSV after every row.
    """
    print(f"--- Stage 2: Visual Refinement & Rendering for user: {user_perspective} ---")
    
    output_dir_path = Path(images_output_dir) / user_perspective
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize columns
    if 'final_prompt' not in df.columns:
        df['final_prompt'] = pd.NA
    if 'img_path' not in df.columns:
        df['img_path'] = pd.NA
    
    current_image_b64 = None
    current_chunk_id = None
    
    # Sort to ensure chronological order
    sorted_df = df.sort_values(by=['chunk_id'], kind='stable')
    
    for index, row in sorted_df.iterrows():
        chunk_id = row['chunk_id']
        
        # Reset context on new chunk
        if chunk_id != current_chunk_id:
            current_image_b64 = None 
            current_chunk_id = chunk_id
            print(f"--- Chunk {current_chunk_id} (Render) ---")

        # Process only specific user rows that have a Stage 1 prompt
        if (row['character'] == user_perspective) and pd.notna(row['initial_prompt']):
            
            initial_prompt = row['initial_prompt']
            img_filename = f"chunk_{chunk_id}_{user_perspective}_{index}.png"
            save_path = os.path.join(output_dir_path, img_filename)
            
            # Skip if Stage 1 was NO_CHANGE
            if initial_prompt == "NO_CHANGE":
                if current_image_b64:
                    # Just carry forward previous image
                    base64_to_pil(current_image_b64).save(save_path)
                    df.at[index, 'img_path'] = str(save_path)
                    df.at[index, 'final_prompt'] = "NO_CHANGE"
                    # Incremental save even on skip/copy
                    df.to_csv(csv_output_path, index=False)
                continue

            # --- STEP A: Visual Refinement (Get Final Prompt) ---
            final_prompt = initial_prompt
            
            # If we have a previous image, ask the VLM to adjust the prompt based on it
            if current_image_b64:
                print(f"  > [{index}] Refining prompt visually...")
                try:
                    # call_prompt_polisher handles the logic: if images are passed, it hits /edit
                    refined_response = await call_prompt_polisher(
                        utterance=initial_prompt,
                        images=[current_image_b64] 
                    )
                    
                    if refined_response and refined_response != "NO_CHANGE":
                        final_prompt = refined_response
                        print(f"  > [{index}] Refined: {final_prompt[:50]}...")
                except Exception as e:
                    print(f"  > [{index}] Refinement failed ({e}). Using initial prompt.")

            # Save final prompt to CSV
            df.at[index, 'final_prompt'] = final_prompt

            # --- STEP B: Image Generation ---
            try:
                print(f"  > [{index}] Generating Image...")
                new_image_b64 = await call_image_gen(
                    final_prompt, 
                    [current_image_b64] if current_image_b64 else None
                )

                if new_image_b64:
                    cleaned_image_pil = clean_image_artifacts(base64_to_pil(new_image_b64))
                    cleaned_image_pil.save(save_path)
                    
                    df.at[index, 'img_path'] = str(save_path)
                    current_image_b64 = pil_to_base64(cleaned_image_pil)
                else:
                    print(f"  > [{index}] Render returned None.")
            except Exception as e:
                print(f"  > [{index}] Render error: {e}")
            
            # --- INCREMENTAL SAVE ---
            # Save after every image generation
            df.to_csv(csv_output_path, index=False)

    return df


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Input CSV file.")
    parser.add_argument("--user", type=str, help="Specific user (optional if using --automatic_users).")
    parser.add_argument("--automatic_users", action='store_true', help="Automatically process all unique characters.")
    parser.add_argument("--create_aug", action='store_true', help="Run Stage 1 (Text -> Prompt).")
    parser.add_argument("--gen_images_from_aug", action='store_true', help="Run Stage 2 (Visual Refinement + Render).")
    parser.add_argument("--aug_output_path", type=str, default="output/")
    parser.add_argument("--images_output_path", type=str, default="output/")
    
    args = parser.parse_args()

    # Server check
    check_servers(args)

    # Setup paths
    input_path = Path(args.file)
    if args.aug_output_path == "output/":
        args.aug_output_path = Path("output") / f"{input_path.stem}_aug{input_path.suffix}"
    
    if args.gen_images_from_aug and args.images_output_path == "output/":
        args.images_output_path = Path("output") / f"{input_path.stem}_images"

    # Load DF
    try:
        # If we are just rendering, try loading the _aug file first if it exists
        if args.gen_images_from_aug and not args.create_aug and Path(args.aug_output_path).exists():
             df = pd.read_csv(args.aug_output_path, dtype={'chunk_id': str})
             print(f"Loaded existing augmented file: {args.aug_output_path}")
        else:
             df = pd.read_csv(args.file, dtype={'chunk_id': str})
             print(f"Loaded input file: {args.file}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Determine Users
    users_to_process = []
    if args.automatic_users:
        # Get all unique characters, filtering out generic system roles if necessary
        users_to_process = [u for u in df['character'].unique() if pd.notna(u)]
        print(f"Automatically detected users: {users_to_process}")
    elif args.user:
        users_to_process = [args.user]
    else:
        print("Error: Must specify --user or --automatic_users")
        return

    # --- Execution Loop ---
    for user in users_to_process:
        print(f"\n==================================================")
        print(f"Processing User: {user}")
        print(f"==================================================")

        if args.create_aug:
            df = await create_aug(
                df=df, 
                user_perspective=user, 
                csv_output_path=str(args.aug_output_path)
            )
            print(f"Finished Stage 1 for {user}. Saved to {args.aug_output_path}")

        if args.gen_images_from_aug:
            df = await gen_images_from_aug(
                df=df, 
                user_perspective=user, 
                images_output_dir=str(args.images_output_path),
                csv_output_path=str(args.aug_output_path)
            )
            print(f"Finished Stage 2 for {user}. Saved to {args.aug_output_path}")

if __name__ == "__main__":
    asyncio.run(main())
