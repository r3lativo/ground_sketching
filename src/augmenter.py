# src/augmenter.py

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import random
from tqdm.asyncio import tqdm

from src.utils import verify_services
from src.data_manager import ConversationDataManager
from src.pipeline import AugmentationPipeline
from src.api_clients import APIClient
from src.mock_api_clients import MockAPIClient  # Mock client for testing
from src.logger import TaskLogger

# Root logger for the augmenter script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [AUGMENTER] - %(message)s')
logger = logging.getLogger("Augmenter")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel Augmentation")
    parser.add_argument("--server_config_path", type=str, default='config/server_config.yaml')
    parser.add_argument("--experiment_config_path", type=str, default='config/experiment_config.yaml')

    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing CSVs")
    parser.add_argument("--output_dir", type=str, default='output/', help="Output root directory")
    parser.add_argument("--n_files", type=int, default=10, help="Maximum number of files to process")
    
    parser.add_argument("--create_aug", action='store_true', help="Run Stage 1: Generate prompts")
    parser.add_argument("--gen_images_from_aug", action='store_true', help="Run Stage 2: Render images")
    parser.add_argument("--relation_triplets", action='store_true', help="Run Stage 3: Create triplets from relations")
    parser.add_argument("--candidate_count", type=int, default=3, help="How many images to generate and check?")

    parser.add_argument("--vlm_concurrency", type=int, default=40, help="VLM Max Concurrent API Requests")
    parser.add_argument("--img_concurrency", type=int, default=8, help="IMG Max Concurrent API Requests")
    parser.add_argument("--oracle", action='store_true', help="Oracle Context Mode")
    
    parser.add_argument("--fake_servers", action='store_true', help="Use Mock Clients but run full pipeline logic")
    
    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    now = datetime.now()
    date_time = now.strftime("%m%d_%H%M%S")

    # 1. Load directories
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir = out_dir / 'fake' if args.fake_servers else out_dir
    out_dir = out_dir / date_time

    if not in_dir.exists():
        logger.error(f"Input directory does not exist: {in_dir}")
        sys.exit(1)

    # 2. Verify Services
    verify_services(args)

    # 3. Discovery
    csv_files = [f for f in in_dir.glob("*.csv") if not f.name.startswith(".")]
    total_files = len(csv_files)
    if not csv_files:
        logger.warning(f"No CSV files found in {in_dir}")
        return
    logger.info(f"Found {total_files} CSV files available.")

    # 4. Initialize Client Strategy
    if args.fake_servers:
        logger.warning("!!! USING FAKE API CLIENTS - NO REAL NETWORK CALLS !!!")
        ClientClass = MockAPIClient
    else:
        ClientClass = APIClient

    # Initialize Resources
    api_client = ClientClass(
        args.server_config_path,
        args.experiment_config_path
    )
    
    pipeline = AugmentationPipeline(api_client)

    # SPLIT SEMAPHORES
    vlm_semaphore = asyncio.Semaphore(args.vlm_concurrency)
    img_semaphore = asyncio.Semaphore(args.img_concurrency)
    
    # 5. Build Tasks (Collect ALL tasks from ALL files first)
    tasks = [] 
    data_managers = []

    # Accumulator for total steps
    total_work_units = 0

    # Select N random files (and discard the rest)
    if 0 < args.n_files < total_files:
        csv_files = random.sample(csv_files, args.n_files)
        logger.info(f"Randomly selected {len(csv_files)} files.")
    else:
        logger.info(f"Processing all {total_files} files.")
    async with api_client:
        for csv_file in csv_files:

            # Build path for each file
            csv_out_dir = out_dir / csv_file.stem

            # Setup Directory Structure
            (csv_out_dir / "logs").mkdir(parents=True, exist_ok=True)
            (csv_out_dir / "traces").mkdir(parents=True, exist_ok=True)
            (csv_out_dir / "images").mkdir(parents=True, exist_ok=True)

            # Output CSV path
            aug_csv_path = csv_out_dir / f"{csv_file.stem}_augmented.csv"
            
            # Create DataManager
            dm = ConversationDataManager(str(csv_file), str(aug_csv_path))
            dm.load_and_prepare()
            dm.save_lock = asyncio.Lock()
            data_managers.append(dm)

            # Calculate how much work this file represents
            # We count 1 unit for every row in every active phase
            rows = len(dm.df)
            file_work = 0
            if args.create_aug: file_work += rows
            if args.gen_images_from_aug: file_work += rows
            if args.relation_triplets: file_work += rows
            
            total_work_units += file_work

            # Get Users
            users = [u for u in dm.df['character'].unique() if isinstance(u, str)]
            logger.info(f"File: {csv_file.name} | Users: {users}")

        # 6. Execute All Tasks with Progress Bar
        # Create the Master Progress Bar
        with tqdm(total=total_work_units, unit="step", desc="Global Progress") as pbar:
            
            for dm in data_managers:
                # We need to re-find users since we are iterating DMs now
                users = [u for u in dm.df['character'].unique() if isinstance(u, str)]
                csv_out_dir = out_dir / Path(dm.source_path).stem 
                
                for user in users:
                    t_logger = TaskLogger(csv_out_dir, Path(dm.source_path).name, user)
                    pipeline_config = {
                        'create': args.create_aug,
                        'render': args.gen_images_from_aug,
                        'oracle': args.oracle,
                        'relation_triplets': args.relation_triplets,
                        'candidate_count': args.candidate_count,
                        'img_output_dir': csv_out_dir / "images"
                    }
                    
                    tasks.append(
                        pipeline.run_single_user(
                            dm, user,
                            vlm_semaphore, img_semaphore,
                            t_logger, pipeline_config,
                            pbar
                        )
                    )

            # Execute
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Error reporting
            for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error(f"Task {i} failed: {res}")

        # 7. Final Save
        logger.info("All tasks done. Performing final save...")
        for dm in data_managers:
            await dm.save()

    logger.info("Experiment Completed Successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
