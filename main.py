# main.py
import argparse
import os
import base64
import csv
import time
from datetime import datetime
import pandas as pd
import yaml
from typing import Optional, Dict, List 

from api_clients import DiffuserApiClient # The client class that handles communication with the generation server
from config_models import ExperimentConfig # The standard configuration for the experiment

class ExperimentRunner:
    """
    Takes care of a complete image generation experiment from start to finish.
    It reads a configuration file, loads data, calls the generation API in a loop,
    and saves the results (images and logs).
    """
    def __init__(self, config_path: str, force_overwrite: bool):
        """
        Initializes the ExperimentRunner.

        Args:
            config_path (str): The file path to the YAML configuration file that defines the experiment.
        """
        # Load the raw YAML data first.
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        self.force_overwrite = force_overwrite
        
        # Parse the raw dictionary into our Pydantic model.
        # This automatically validates, applies defaults, and gives clear errors.
        self.config: ExperimentConfig = ExperimentConfig(**raw_config)
        print(f"Loaded and validated experiment: {self.config.experiment_name}")
        
        # Initialize the API client with the server URL from the config.
        self.client = DiffuserApiClient(self.config.api_url)
        
        # This state variable holds the most recently generated image (as a Base64 string).
        # It is used as the input for the next step in 'utterance' mode. It's None at the start.
        self.current_image_b64: Optional[str] = None

        os.makedirs(os.path.dirname(self.config.output_dir), exist_ok=True)


    def _get_output_path(self, gen_step: str) -> str:
        """Generates a standardized, deterministic file path for an output image."""
        filename = f"{self.config.experiment_name}_{gen_step}.png"
        return os.path.join(self.config.output_dir, filename)

    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the dialogue data from the CSV file specified in the config,
        and filters it to get only the rows relevant to this experiment.
        
        Returns:
            A pandas DataFrame containing the filtered dialogue lines, or None if an error occurs.
        """
        try:
            df = pd.read_csv(self.config.csv_file_path, dtype={'chunk_id': str})
            # Filter the DataFrame to include only rows matching the chunk_id AND character from the config.
            perspective_df = df[(df['chunk_id'] == self.config.chunk_id) & (df['character'] == self.config.character)]
            if perspective_df.empty:
                print(f"No lines found for character '{self.config.character}' in chunk '{self.config.chunk_id}'.")
                return None
            return perspective_df
        except FileNotFoundError:
            print(f"Error: The file '{self.config.csv_file_path}' was not found.")
            return None
        except KeyError as e:
            print(f"Error: Missing required key in CSV file: {e}")
            return None

    def _log_result(self, gen_step: str, final_prompt: str, negative_prompt:str, filename: str, latency: float, num_steps: int, success: bool = True):
        """
        Appends a new row to the experiment's log CSV file with details about a single generation step.

        Args:
            gen_step (str): A descriptive name for the step (e.g., "chunk_0", "u1_of_10").
            final_prompt (str): The actual prompt sent to the image model (could be raw or refined).
            negative_prompt (str): The negative prompt used for this generation.
            filename (str): The name of the output image file.
            latency (float): The time taken for the API call, in seconds.
            num_steps (int): The number of inference steps used for this generation.
            success (bool): Whether the process was a success or it failed.
        """
        log_file = self.config.log_file_path
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # If the file is new, write the header row first.
            if not os.path.isfile(log_file):
                writer.writerow([
                    "timestamp", "experiment_name", "status", "mode",
                    "chunk_id", "character", "gen_step",
                    "final_prompt", "negative_prompt", "output_file",
                    "latency_s", "num_inference_steps"
                ])

            # Write the data for the current generation step.
            writer.writerow([
                datetime.now().isoformat(), self.config.experiment_name,
                "SUCCESS" if success else "FAILURE", self.config.mode,
                self.config.chunk_id, self.config.character, gen_step,
                final_prompt, negative_prompt, filename,
                f"{latency:.2f}", num_steps
            ])

    def run(self):
        """Main execution method, now with resumability logic."""
        dialogue_df = self._load_data()
        if dialogue_df is None or dialogue_df.empty:
            print("Aborting experiment due to data loading issues.")
            return

        # Branch the execution logic based on the 'mode' specified in the config file.
        if self.config.mode == 'chunk':
            gen_step_name = "chunk_0"
            output_path = self._get_output_path(gen_step_name)
            
            # --- RESUMABILITY LOGIC ---
            if os.path.exists(output_path) and not self.force_overwrite:
                print(f"Output for {gen_step_name} already exists. Skipping.")
                return # The whole experiment is just one step, so we can exit.
            # --- END OF RESUMABILITY LOGIC ---

            dialogue_lines = dialogue_df['text'].tolist()
            self._run_single_generation(dialogue_lines, 0, gen_step_name)
        
        # In 'utterance' mode, we loop through each dialogue line one by one.
        elif self.config.mode == 'utterance':
            total_steps = len(dialogue_df)
            for idx, row in enumerate(dialogue_df.itertuples()):
                gen_step_name = f"u{idx}_of_{total_steps-1}"
                output_path = self._get_output_path(gen_step_name)

                # --- RESUMABILITY LOGIC ---
                if os.path.exists(output_path) and not self.force_overwrite:
                    print(f"Skipping step {gen_step_name}, output already exists.")
                    # IMPORTANT: Load the existing image to maintain the chain.
                    with open(output_path, "rb") as f:
                        self.current_image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    continue # Move to the next iteration of the loop.
                # --- END OF RESUMABILITY LOGIC ---

                dialogue_lines = [row.text]
                self._run_single_generation(dialogue_lines, idx, gen_step_name)
                
                if self.current_image_b64 is None:
                    print(f"Stopping utterance sequence due to generation error at step {gen_step_name}.")
                    break

    def _run_single_generation(self, dialogue_lines: List[str], idx: int, gen_step: str):
        """
        Handles the logic for a single API call: prepares the payload, sends the request,
        and processes the response.

        Args:
            dialogue_lines (List[str]): The dialogue line(s) for this step.
            idx (int): The index of the current step (0 for the first, 1+ for subsequent).
            gen_step (str): A descriptive name for the generation step (for logging).
        """
        # Pydantic model provides the full parameters object with defaults applied.
        current_params = self.config.parameters.model_dump()
        if idx == 0:
            num_steps = self.config.parameters.initial_num_steps
            print(f"   Using initial steps: {num_steps}")
        else:
            num_steps = self.config.parameters.num_inference_steps
        
        current_params['num_inference_steps'] = num_steps
        
        payload = {
            "dialogue_lines": dialogue_lines,
            "refine_prompt": self.config.refine_prompt,
            "refiner_template_name": self.config.refiner_template_name,
            "initial_image_b64": self.current_image_b64,
            "parameters": current_params
        }

        print(f"\n--- Running Step: {gen_step} (Refine: {payload['refine_prompt']}) ---")
        
        start_time = time.time()
        result = self.client.generate(payload)
        latency = time.time() - start_time

        if result:
            # --- Success Path ---
            final_prompt, result_b64 = result
            output_path = self._get_output_path(gen_step)
            
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            print(f"   Image saved to {output_path} (Latency: {latency:.2f}s)")
            
            self.current_image_b64 = result_b64

            # Log the successful result
            self._log_result(
                gen_step=gen_step,
                final_prompt=final_prompt,
                negative_prompt=current_params.get("negative_prompt"),
                filename=os.path.basename(output_path),
                latency=latency,
                num_steps=num_steps,
                success=True
            )
        else:
            # --- Failure Path ---
            print("   Generation failed. Received no image from the server.")
            self.current_image_b64 = None

            # Log the failure with placeholder values
            self._log_result(
                gen_step=gen_step,
                final_prompt="GENERATION FAILED",
                negative_prompt=current_params.get("negative_prompt"),
                filename="N/A",
                latency=latency,
                num_steps=num_steps,
                success=False
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scene generation experiments from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment's YAML configuration file.")
    parser.add_argument("--force", action="store_true", help="If specified, overwrite existing images and re-run all generation steps.")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: Experiment configuration file not found at {args.config}")
    else:
        try:
            runner = ExperimentRunner(config_path=args.config, force_overwrite=args.force)
            runner.run()
        except Exception as e:
            print(f"An error occurred during experiment setup or execution: {e}")
