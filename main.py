# main.py
import argparse
import os
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
from typing import Optional
from generators import DiffuserGeneratorClient, SgpGeneratorClient

# --- Configuration ---
CSV_FILE_PATH = "conversation.csv"
GENERATOR_APIS = {
    "diffuser": "http://127.0.0.1:8000/generate_diff",
    "sgp"     : "http://127.0.0.1:8001/generate_svg"
}

class ExperimentRunner:
    def __init__(self, generator_choice: str, positive_magic: str):
        api_url = GENERATOR_APIS.get(generator_choice)
        if not api_url:
            raise ValueError(f"Invalid generator choice: {generator_choice}. Must be one of {list(GENERATOR_APIS.keys())}")

        if generator_choice == "diffuser":
            self.chosen_g = "diffuser"
            self.generator = DiffuserGeneratorClient(api_url)
        else:
            self.chosen_g = "sgp"
            self.generator = SgpGeneratorClient(api_url)

        self.positive_magic = positive_magic
        self.current_image_b64: Optional[str] = None

    def _load_data(self, chunk_id: str, character: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(CSV_FILE_PATH, dtype={'chunk_id': str})
            perspective_df = df[(df['chunk_id'] == chunk_id) & (df['character'] == character)]
            if perspective_df.empty:
                print(f"No lines found for '{character}' in chunk '{chunk_id}'.")
                return None
            return perspective_df
        except FileNotFoundError:
            print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
            return None

    def run_chunk_mode(self, chunk_id: str, character: str, args):
        df = self._load_data(chunk_id, character)
        if df is None: return

        full_description = ' '.join(df['character'] + ': ' + df['text'])
        prompt = f"A first-person point-of-view shot visualizing the scene from this conversation: '{full_description}'\n{self.positive_magic}"

        if self.chosen_g == "sgp": prompt = "Please write SVG code for generating the image corresponding to the following description: " + prompt

        payload = {
            "prompt": prompt, "initial_image_b64": None, "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.steps, "canvas_height": args.canvas_size
        }
        
        print(f"\nSending consolidated prompt for chunk {chunk_id}...")
        image_b64 = self.generator.generate(payload)
        
        if image_b64:
            self._save_image(image_b64, f"{character.lower()}_chunk_{chunk_id}.png")

    def run_utterance_mode(self, chunk_id: str, character: str, args):
        df = self._load_data(chunk_id, character)
        if df is None: return

        for idx, row in enumerate(df.itertuples()):
            line = f"{row.character}: {row.text}"
            prompt_action = "visualizing the scene" if idx == 0 else "modifying the scene"
            prompt = f"A first-person point-of-view shot, {prompt_action} from this utterance: '{line}'\n{self.positive_magic}"
            if self.chosen_g == "sgp": prompt = "Please write SVG code for generating the image corresponding to the following description: " + prompt

            payload = {
                "prompt": prompt, "initial_image_b64": self.current_image_b64, "negative_prompt": args.negative_prompt,
                "num_inference_steps": args.steps, "canvas_height": args.canvas_size
            }

            print(f"\nSending prompt for utterance {idx + 1}...")
            image_b64 = self.generator.generate(payload)

            if image_b64:
                self.current_image_b64 = image_b64
                self._save_image(image_b64, f"{character.lower()}_utt_{idx}.png")
            else:
                print("Stopping due to API error.")
                break

    def _save_image(self, b64_string: str, filename: str):
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", filename)
        img_data = base64.b64decode(b64_string)
        Image.open(BytesIO(img_data)).save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene generation experiments.")
    parser.add_argument("--generator", type=str, required=True, choices=["diffuser", "sgp"], help="Generator backend to use.")
    parser.add_argument("--mode", type=str, required=True, choices=["chunk", "utterance"], help="Prompting mode.")
    parser.add_argument("--chunk_id", type=str, required=True, help="Target chunk ID from the CSV.")
    parser.add_argument("--character", type=str, required=True, help="Target character POV.")
    parser.add_argument("--canvas_size", type=int, default=512, help="Canvas size for the image.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--negative_prompt", type=str, default="text, people, blurry, low quality", help="Negative prompt.")
    parser.add_argument("--style", type=str, default="Style: detailed sketch, composition, simple background", help="Positive style keywords.")
    args = parser.parse_args()

    runner = ExperimentRunner(args.generator, args.style)
    
    if args.mode == 'chunk':
        runner.run_chunk_mode(args.chunk_id, args.character, args)
    else:
        runner.run_utterance_mode(args.chunk_id, args.character, args)
