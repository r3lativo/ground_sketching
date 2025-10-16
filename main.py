# main.py
import argparse
import os
import base64
import pandas as pd
import yaml
import csv
from io import BytesIO
from PIL import Image
from typing import Optional, Dict
from datetime import datetime
import time

from generators import DiffuserGeneratorClient, SgpGeneratorClient, RefinerClient

class ExperimentRunner:
    def __init__(self, config: Dict, generator_choice: str):
        self.config = config
        self.generator_choice = generator_choice
        self.gen_config = config[generator_choice]
        self.exp_config = config['experiment_settings']

        if generator_choice == "diffuser":
            self.generator = DiffuserGeneratorClient(self.gen_config['api_url'])
        else:
            self.generator = SgpGeneratorClient(self.gen_config['api_url'])
        
        self.refiner = RefinerClient(config['sgp']['refiner_api_url'])
        
        self.current_image_b64: Optional[str] = None
        self.current_svg_text: Optional[str] = None

    def _load_data(self, chunk_id: str, character: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.exp_config['csv_file_path'], dtype={'chunk_id': str})
            perspective_df = df[(df['chunk_id'] == chunk_id) & (df['character'] == character)]
            if perspective_df.empty:
                print(f"No lines found for '{character}' in chunk '{chunk_id}'.")
                return None
            return perspective_df
        except FileNotFoundError:
            print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
            return None
        return df[(df['chunk_id'] == chunk_id) & (df['character'] == character)]

    def _log_result(self, args: argparse.Namespace, prompt: str, filename: str, latency: float, gen_step: str, num_inference_steps: int):
        log_file = self.exp_config['log_file_path']
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "generator", "mode", "prompt_mode", "chunk_id", "character", "gen_step", "prompt", "output_file", "latency_s", "num_inference_steps"])
            
            writer.writerow([
                datetime.now().isoformat(), args.generator, args.mode, args.prompt_mode, args.chunk_id,
                args.character, gen_step, prompt, filename, f"{latency:.2f}", num_inference_steps
            ])

    def _get_description(self, prompt_mode: str, dialogue_df: pd.DataFrame) -> str:
        #raw_lines = (dialogue_df['character'] + ': ' + dialogue_df['text']).tolist()
        raw_lines = dialogue_df['text'].tolist()  # Character removed totally from text to be processed
        if prompt_mode == 'refined':
            print("Refining description from dialogue...")
            template = self.config['refiner']['prompt_template']
            return self.refiner.refine(raw_lines, template)
        return ' '.join(raw_lines)

    def run(self, args: argparse.Namespace):
        df = self._load_data(args.chunk_id, args.character)
        if df is None or df.empty:
            print("No data found for the specified criteria.")
            return

        if args.mode == 'chunk':
            description = self._get_description(args.prompt_mode, df)
            self._run_single_generation(description, args, "chunk_0")
        else: # utterance mode
            total_len = len(df)
            for idx, row in enumerate(df.itertuples()):
                
                # IF IMG ALREADY THERE, LOAD IT AND SKIP TO THAT STEP
                if idx < args.start_idx:
                    continue
                if args.initial_file and os.path.isfile(args.initial_file):
                    with open(args.initial_file, "rb") as f:
                        self.current_image_b64 = base64.b64encode(f.read())

                description = self._get_description(args.prompt_mode, pd.DataFrame([row]))
                self._run_single_generation(description, args, f"{idx+1}_of_{total_len}", f"u{idx}")
                if self.current_image_b64 is None and self.current_svg_text is None:
                    print("Stopping due to generation error in utterance mode.")
                    break
    
    def _run_single_generation(self, description: str, args: argparse.Namespace, gen_step: str, u_idx: str):
        params = self.gen_config['parameters']
        templates = params['prompt_templates']
        
        is_first_step = self.current_image_b64 is None and self.current_svg_text is None
        prompt_template = templates['initial'] if is_first_step else templates[u_idx]

        prompt = prompt_template.format(
            description=description,
            positive_magic=params.get('positive_magic', ''),
            previous_svg=self.current_svg_text or ''
        ).strip()

        num_inference_steps = params.get('steps')

        payload = {
            "prompt": prompt,
            "negative_prompt": params.get('negative_prompt'),
            "num_inference_steps": num_inference_steps,
            "canvas_height": params.get('canvas_size'),
            "initial_image_b64": self.current_image_b64,
        }

        print(f"\n--- Running Step: {gen_step} ---")
        start_time = time.time()
        result = self.generator.generate(payload)
        latency = time.time() - start_time

        if result:
            filename = f"{args.generator}_{args.mode[:3]}_{args.prompt_mode[:3]}_{args.character.lower()}_{args.chunk_id}_{gen_step}_{num_inference_steps}.png"
            output_path = os.path.join(self.exp_config['output_dir'], filename)
            os.makedirs(self.exp_config['output_dir'], exist_ok=True)
            
            image_b64 = result if isinstance(result, str) else result['image_b64']
            
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            print(f"Image saved to {output_path}")

            # Update state for next iteration
            self.current_image_b64 = image_b64
            if self.generator_choice == 'sgp' and isinstance(result, dict):
                self.current_svg_text = result['svg_text']

            self._log_result(args, prompt, filename, latency, gen_step, num_inference_steps)
        else:
            self.current_image_b64 = None
            self.current_svg_text = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene generation experiments.")
    parser.add_argument("--generator", type=str, required=True, choices=["diffuser", "sgp"], help="Generator backend to use.")
    parser.add_argument("--mode", type=str, required=True, choices=["chunk", "utterance"], help="Prompting mode.")
    parser.add_argument("--prompt_mode", type=str, default="raw", choices=["raw", "refined"], help="Whether to use raw dialogue or a refined description.")
    parser.add_argument("--chunk_id", type=str, required=True, help="Target chunk ID from the CSV.")
    parser.add_argument("--character", type=str, required=True, help="Target character POV.")
    parser.add_argument("--initial_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    runner = ExperimentRunner(config, args.generator)
    runner.run(args)
