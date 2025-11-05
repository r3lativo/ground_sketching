import pandas as pd
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from PIL import Image
import os

from src.utils import pil_to_base64, base64_to_pil, clean_image_artifacts
from src.api_clients import (
    call_contextual_prompt_polisher,
    call_contextual_edit_prompt_polisher,
    call_image_gen
)

async def create_prompts(df: pd.DataFrame, user_perspective: str) -> pd.DataFrame:
    """
    Generates 'prompt_to_render' using the new contextual logic,
    passing a history of previous prompts instead of images.
    """
    print(f"Creating contextual prompts for user: {user_perspective}...")

    df['prompt_to_render'] = pd.NA
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
            polished_prompt = None

            try:
                if not previous_prompts_history:
                    # --- This is the FIRST utterance in the chunk ---
                    # call_contextual_prompt_polisher(context, utterance)
                    print(f"  > Index {index} (Create): Polishing first utterance...")
                    polished_prompt = await call_contextual_prompt_polisher(
                        context=context_list,
                        utterance=utterance_text
                    )
                    print(f"  > {polished_prompt}")
                else:
                    # --- This is a SUBSEQUENT utterance in the chunk ---
                    # call_contextual_edit_prompt_polisher(context, utterance, previous prompts)
                    print(f"  > Index {index} (Edit): Polishing edit utterance...")
                    polished_prompt = await call_contextual_edit_prompt_polisher(
                        context=context_list,
                        utterance=utterance_text,
                        previous_prompts=previous_prompts_history
                    )
                    print(f"  > {polished_prompt}")
                # --- Save the result ---
                if polished_prompt:
                    df.at[index, 'prompt_to_render'] = polished_prompt
                    # Add this *new* prompt to the history for the *next* iteration
                    previous_prompts_history.append(polished_prompt)
                else:
                    print(f"Warning: Polishing failed for index {index}. Skipping.")
            
            except Exception as e:
                print(f"Error during prompt creation at index {index}: {e}")
                # We don't add to history if it failed
                pass
                
    print("Contextual prompt creation complete.")
    return df


async def render_prompts(
    df: pd.DataFrame, 
    user_perspective: str, 
    render_output_dir: str
) -> pd.DataFrame:
    """
    (Unchanged)
    Generates images by iterating through prompts sequentially,
    passing the output of one step as the input to the next.
    """
    print(f"Rendering prompts for user: {user_perspective}...")
    
    output_path = Path(render_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_path.resolve()}")

    df['img_path'] = pd.NA
    
    current_image_b64 = None
    current_chunk_id = None
    
    sorted_df = df.sort_values(by=['chunk_id'], kind='stable')
    
    for index, row in sorted_df.iterrows():
        chunk_id = row['chunk_id']
        
        if chunk_id != current_chunk_id:
            current_image_b64 = None 
            current_chunk_id = chunk_id
            print(f"--- Processing Chunk {current_chunk_id} (Render) ---")

        if (row['character'] == user_perspective) and pd.notna(row['prompt_to_render']):
            prompt = row['prompt_to_render']
            
            img_filename = f"chunk_{chunk_id}_index_{index}.png"
            save_path = os.path.join(output_path, img_filename)
            
            try:
                if current_image_b64:
                    new_image_b64 = await call_image_gen(prompt, [current_image_b64])
                else:
                    new_image_b64 = await call_image_gen(prompt, None)

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
                    # Keep old image as context
            except Exception as e:
                print(f"Error rendering prompt at index {index}: {e}")
                pass

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
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the output conversation_aug.csv file. Defaults to [input_file]_aug.csv if not provided."
    )
    parser.add_argument(
        "--user", 
        type=str, 
        required=True, 
        help="The 'character' (user) perspective to render (e.g., 'Nathan')."
    )
    parser.add_argument(
        "--create_prompts", 
        action='store_true', 
        help="Flag to generate the 'prompt_to_render' column."
    )
    parser.add_argument(
        "--render_prompts", 
        action='store_true', 
        help="Flag to render images from the 'prompt_to_render' column."
    )
    parser.add_argument(
        "--render_output", 
        type=str, 
        default="output/conversation_images",
        help="Directory to save rendered images. Used if --render_prompts is set."
    )
    
    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.file)
        new_filename = f"{input_path.stem}_aug{input_path.suffix}"
        args.output = str(input_path.with_name(new_filename))
        print(f"No --output provided. Defaulting to: {args.output}")

    if not args.create_prompts and not args.render_prompts:
        print("Neither --create_prompts nor --render_prompts was set. Doing nothing.")
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

    if args.create_prompts:
        df = await create_prompts(df, args.user)
        df.to_csv(args.output, index=False)
        print(f"Saved augmented CSV with prompts to {args.output}")

    if args.render_prompts:
        if not args.create_prompts:
            try:
                df = pd.read_csv(args.output)
                print(f"Loaded {args.output} for rendering.")
                if 'prompt_to_render' not in df.columns:
                    print(f"Error: {args.output} does not have 'prompt_to_render' column.")
                    return
            except FileNotFoundError:
                print(f"Error: {args.output} not found. Run with --create_prompts first.")
                return

        df = await render_prompts(df, args.user, args.render_output)
        df.to_csv(args.output, index=False)
        print(f"Saved final CSV with image paths to {args.output}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
