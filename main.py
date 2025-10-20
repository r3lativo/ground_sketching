# run_experiment.py
import argparse
import os
import base64
import pandas as pd
import yaml
import csv
import time
from datetime import datetime
from typing import Optional, Dict, List

from api_clients import DiffuserApiClient

class ExperimentRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config: Dict = yaml.safe_load(f)
        
        print(f"Loaded experiment: {self.config['experiment_name']}")
        self.client = DiffuserApiClient(self.config['api_url'])
        self.current_image_b64: Optional[str] = None

    def _load_data(self) -> Optional[pd.DataFrame]:
        try:
            exp_conf = self.config
            df = pd.read_csv(exp_conf['csv_file_path'], dtype={'chunk_id': str})
            perspective_df = df[(df['chunk_id'] == exp_conf['chunk_id']) & (df['character'] == exp_conf['character'])]
            if perspective_df.empty:
                print(f"No lines found for character '{exp_conf['character']}' in chunk '{exp_conf['chunk_id']}'.")
                return None
            return perspective_df
        except FileNotFoundError:
            print(f"Error: The file '{exp_conf['csv_file_path']}' was not found.")
            return None
        except KeyError as e:
            print(f"Error: Missing required key in CSV file: {e}")
            return None

    def _log_result(self, gen_step: str, final_prompt: str, negative_prompt:str, filename: str, latency: float, num_steps: int):
        log_file = self.config['log_file_path']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write the headers
                writer.writerow([
                    "timestamp", "experiment_name", "mode",
                    "chunk_id", "character", "gen_step",
                    "final_prompt", "negative_prompt", "output_file",
                    "latency_s", "num_inference_steps"
                ])
            # Add actual content
            writer.writerow([
                datetime.now().isoformat(), self.config['experiment_name'], self.config['mode'],
                self.config['chunk_id'], self.config['character'], gen_step,
                final_prompt, negative_prompt, filename,
                f"{latency:.2f}", num_steps
            ])

    def run(self):
        """Main execution method to run the experiment."""
        dialogue_df = self._load_data()
        if dialogue_df is None or dialogue_df.empty:
            print("Aborting experiment due to data loading issues.")
            return

        if self.config['mode'] == 'chunk':
            dialogue_lines = dialogue_df['text'].tolist()
            # For chunk mode, the index is always 0 (it's the first and only step)
            self._run_single_generation(dialogue_lines, 0, "chunk_0")
        
        elif self.config['mode'] == 'utterance':
            total_steps = len(dialogue_df)
            for idx, row in enumerate(dialogue_df.itertuples()):
                dialogue_lines = [row.text] # Send as a list with one item
                gen_step_name = f"u{idx}_of_{total_steps-1}"

                self._run_single_generation(dialogue_lines, idx, gen_step_name)
                if self.current_image_b64 is None:
                    print(f"Stopping utterance sequence due to generation error at step {gen_step_name}.")
                    break

    def _run_single_generation(self, dialogue_lines: List[str], idx: int, gen_step: str):
        """
        Prepares and sends a request, dynamically setting the number of inference steps.

        Args:
            dialogue_lines (List[str]): The dialogue for this step.
            idx (int): The index of the current step (0 for the first, 1+ for subsequent).
            gen_step (str): A descriptive name for the generation step (for logging).
        """
        # --- DYNAMIC PARAMETER LOGIC ---
        # Make a copy of the parameters to avoid modifying the original config dict
        current_params = self.config.get("parameters", {}).copy()
        
        # Check if we are on the first step (idx == 0)
        if idx == 0 and 'initial_num_steps' in current_params:
            # Use the initial, higher step count
            num_steps = current_params['initial_num_steps']
            print(f"   Using initial steps: {num_steps}")
        else:
            # Use the standard, lower step count for subsequent steps
            num_steps = current_params.get('num_inference_steps', 30) # Default to 30 if not specified
        
        # Set the final 'num_inference_steps' in our temporary params dictionary
        current_params['num_inference_steps'] = num_steps
        # --- END OF DYNAMIC LOGIC ---

        # Construct the payload for the new /generate endpoint
        payload = {
            "dialogue_lines": dialogue_lines,
            "refine_prompt": self.config.get("refine_prompt", False),
            "refiner_template_name": self.config.get("refiner_template_name", "default"),
            "initial_image_b64": self.current_image_b64,
            "parameters": current_params # Use the modified parameters
        }

        print(f"\n--- Running Step: {gen_step} (Refine: {payload['refine_prompt']}) ---")
        
        start_time = time.time()
        result = self.client.generate(payload)
        latency = time.time() - start_time

        if result:
            final_prompt, result_b64 = result  # Unpack the valid result
            
            filename = f"{self.config['experiment_name']}_{gen_step}.png"
            output_path = os.path.join(self.config['output_dir'], filename)
            os.makedirs(os.path.dirname(self.config['output_dir']), exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            print(f"   Image saved to {output_path} (Latency: {latency:.2f}s)")

            self.current_image_b64 = result_b64
            self._log_result(gen_step, final_prompt, current_params.get("negative_prompt"), filename, latency, num_steps)
        else:
            print("   Generation failed. Received no image from the server.")
            self.current_image_b64 = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene generation experiments from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment's YAML configuration file.")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: Experiment configuration file not found at {args.config}")
    else:
        runner = ExperimentRunner(config_path=args.config)
        runner.run()
