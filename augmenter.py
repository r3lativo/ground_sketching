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
    Checks if all required backend services are running before starting.
    Provides specific error messages for failed services.
    """
    server_config = load_config(config_path='config/server_config.yaml')
    failed_services = []

    # --- Check Prompt Polisher (Always required) ---
    polisher_conf = server_config["vlm_service"]
    if not check_server(polisher_conf["gateway"]["host"], polisher_conf["gateway"]["port"]):
        failed_services.append(
            f"Prompt Polisher service at http://{polisher_conf["gateway"]['host']}:{polisher_conf["gateway"]['port']}"
        )

    # --- Check Image Gen (Conditionally required) ---
    if args.gen_images_from_aug:  # Check if we intend to render
        img_gen_conf = server_config["image_gen_service"]
        if not check_server(img_gen_conf["host"], img_gen_conf["port"]):
            failed_services.append(
                f"Image Gen service at http://{img_gen_conf['host']}:{img_gen_conf['port']}"
            )

    # --- Report results ---
    if failed_services:
        print("\n[Error] One or more required services are down:")
        for service_msg in failed_services:
            print(f"- {service_msg}")
        
        print("Exiting.")
        sys.exit(1)
    
    # If we get here, all required services are up.
    print("\nAll required services are running.")
        

async def create_aug(df: pd.DataFrame, user_perspective: str) -> pd.DataFrame:
    """
    Generates 'img_gen_prompt',
    passing a history of previous prompts instead of images.
    """
    print(f"Creating contextual prompts for user: {user_perspective}...")

    df['img_gen_prompt'] = pd.NA
    grouped = df.groupby('chunk_id')

    for chunk_id, chunk_df in grouped:
        print(f"--- Processing Chunk {chunk_id} (Create Prompts) ---")
        
        # This list tracks previous prompts *within this chunk*
        previous_prompts_history = []
        
        # Get all utterances for the specified user
        user_utterances = chunk_df[
            chunk_df['character'] == user_perspective
        ].sort_index()

        # Get full chunk context (all text from all users in order)
        context_list = chunk_df.sort_index()['text'].tolist()
        
        for index, row in user_utterances.iterrows():
            utterance_text = row['text']
            
            # Log which type of polishing we're doing
            if not previous_prompts_history:
                print(f"  > Index {index} (Create): Polishing first utterance...")
            else:
                print(f"  > Index {index} (Edit): Polishing edit utterance...")

            try:
                polished_prompt = await call_prompt_polisher(
                    utterance=utterance_text,
                    context=context_list,
                    previous_prompts=previous_prompts_history if previous_prompts_history else None
                )

                print(f"  > {polished_prompt}")
                
                # --- Save the result ---
                if polished_prompt:
                    df.at[index, 'img_gen_prompt'] = polished_prompt
                    # Add this *new* prompt to the history for the *next* iteration if it is NOT `NO_CHANGE`
                    if polished_prompt != "NO_CHANGE":
                            previous_prompts_history.append(polished_prompt)
                else:
                    print(f"Warning: Polishing failed for index {index}. Skipping.")
            
            except Exception as e:
                print(f"Error during prompt creation at index {index}: {e}")
                # We don't add to history if it failed
                pass
                
    print("Contextual prompt creation complete.")
    return df


async def gen_images_from_aug(
    df: pd.DataFrame, 
    user_perspective: str, 
    images_output_path_dir: str
) -> pd.DataFrame:
    """
    Generates images by iterating through prompts sequentially,
    passing the output of one step as the input to the next.
    """
    print(f"Rendering prompts for user: {user_perspective}...")
    
    output_path = Path(images_output_path_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_path.resolve()}")

    df['img_path'] = pd.NA
    
    current_image_b64 = None
    current_chunk_id = None
    
    # Ensure processing in correct order
    sorted_df = df.sort_values(by=['chunk_id', 'index'], kind='stable')
    
    for index, row in sorted_df.iterrows():
        chunk_id = row['chunk_id']
        
        # Reset the image context for each new chunk
        if chunk_id != current_chunk_id:
            current_image_b64 = None 
            current_chunk_id = chunk_id
            print(f"--- Processing Chunk {current_chunk_id} (Render) ---")

        # Only render for the target user and if a prompt exists
        if (row['character'] == user_perspective) and pd.notna(row['img_gen_prompt']):
            prompt = row['img_gen_prompt']
            
            img_filename = f"chunk_{chunk_id}_index_{index}.png"
            save_path = os.path.join(output_path, img_filename)
            
            # Check whether to skip rendering if `NO_CHANGE`
            if prompt == "NO_CHANGE":
                print(f"Image for index:{index} has NO_CHANGE. Re-using previous image.")
                if current_image_b64:
                    base64_to_pil(current_image_b64).save(save_path)
                    df.at[index, 'img_path'] = str(save_path)
                continue # Skip rendering

            try:
                # Pass the previous image (or None) to the image gen service
                new_image_b64 = await call_image_gen(
                    prompt, 
                    [current_image_b64] if current_image_b64 else None
                )

                if new_image_b64:
                    # convert to pil and clean
                    cleaned_image_pil = clean_image_artifacts(base64_to_pil(new_image_b64))
                    # save
                    cleaned_image_pil.save(save_path)
                    print(f"Image for index:{index} saved at {save_path}")
                    df.at[index, 'img_path'] = str(save_path)
                    # convert back to base64 and pass
                    current_image_b64 = pil_to_base64(cleaned_image_pil)
                else:
                    print(f"Warning: Render function returned None for index {index}")
                    # Keep old image as context if render fails
            except Exception as e:
                print(f"Error rendering prompt at index {index}: {e}")
                pass # Keep old image and continue

    print("Image rendering complete.")
    return df


async def main():
    parser = argparse.ArgumentParser(
        description="Augment conversation CSVs with generative prompts and render them."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Path to the input conversation.csv file."
    )
    parser.add_argument(
        "--user", 
        type=str, 
        required=True, 
        help="The 'character' (user) perspective to render (e.g., 'Nathan')."
    )
    parser.add_argument(
        "--create_aug", 
        action='store_true', 
        help="Flag to generate the 'img_gen_prompt' column."
    )
    parser.add_argument(
        "--aug_output_path", 
        type=str, 
        default="output/",  # Changed default
        help="Path to save the output conversation_aug.csv file. Defaults to output/[input_file_stem]_aug.csv if not provided."
    )
    parser.add_argument(
        "--gen_images_from_aug", 
        action='store_true', 
        help="Flag to render images from the 'img_gen_prompt' column."
    )
    parser.add_argument(
        "--images_output_path", 
        type=str, 
        default="output/",  # Changed default
        help="Directory to save rendered images. Defaults to output/[input_file_stem]_aug/ if not provided."
    )
    
    args = parser.parse_args()

    # Call the improved server check
    check_servers(args)

    # Set the default output path for augmented version if it is not provided
    # Check if the path is still the default directory value
    if args.aug_output_path == "output/":
        input_path = Path(args.file)
        new_filename = f"{input_path.stem}_aug{input_path.suffix}"
        args.aug_output_path = Path("output") / new_filename
        print(f"No --aug_output_path provided. Defaulting to: {args.aug_output_path}")

    # If generating images but no output path for those, set it
    # Check if the path is still the default directory value
    if args.gen_images_from_aug and args.images_output_path == "output/":
        input_path = Path(args.file)
        new_directory = f"{input_path.stem}_aug"
        args.images_output_path = Path("output") / new_directory
        print(f"No --images_output_path provided. Defaulting to: {args.images_output_path}")

    if not args.create_aug and not args.gen_images_from_aug:
        print("Neither --create_aug nor --gen_images_from_aug was set. Doing nothing.")
        return

    try:
        df = pd.read_csv(args.file, dtype={'chunk_id': str})
        print(f"Loaded {args.file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.file}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if args.create_aug:
        df = await create_aug(df=df, user_perspective=args.user)
        df.to_csv(args.aug_output_path, index=False)
        print(f"Saved augmented CSV with prompts to {args.aug_output_path}")

    if args.gen_images_from_aug:
        if not args.create_aug:
            # If only rendering, load the file that supposedly has prompts
            try:
                # Use the path that was potentially set by the default logic
                aug_file_to_load = args.aug_output_path
                
                if args.aug_output_path == "output/":
                     input_path = Path(args.file)
                     new_filename = f"{input_path.stem}_aug{input_path.suffix}"
                     aug_file_to_load = Path("output") / new_filename
                
                df = pd.read_csv(aug_file_to_load)
                print(f"Loaded {aug_file_to_load} for rendering.")
                if 'img_gen_prompt' not in df.columns:
                    print(f"Error: {aug_file_to_load} does not have 'img_gen_prompt' column.")
                    return
            except FileNotFoundError:
                print(f"Error: {aug_file_to_load} not found. Run with --create_aug first.")
                return

        df = await gen_images_from_aug(df=df, user_perspective=args.user, images_output_path_dir=args.images_output_path)
        df.to_csv(args.aug_output_path, index=False)
        print(f"Saved final CSV with image paths to {args.aug_output_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
