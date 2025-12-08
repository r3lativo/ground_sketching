# augmenter.py

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.utils import setup_logging, check_server, load_config
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
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing CSVs")
    parser.add_argument("--output_dir", type=str, default='output/', help="Output root directory")
    
    parser.add_argument("--create_aug", action='store_true', help="Run Stage 1: Generate prompts")
    parser.add_argument("--gen_images_from_aug", action='store_true', help="Run Stage 2: Render images")
    parser.add_argument("--candidate_count", type=int, default=3, help="How many images to generate and check?")

    parser.add_argument("--vlm_concurrency", type=int, default=50, help="VLM Max Concurrent API Requests")
    parser.add_argument("--img_concurrency", type=int, default=8, help="IMG Max Concurrent API Requests")
    parser.add_argument("--oracle", action='store_true', help="Oracle Context Mode")
    
    parser.add_argument("--fake_servers", action='store_true', help="Use Mock Clients but run full pipeline logic")
    
    return parser.parse_args()

def verify_services(args):
    """Checks if required servers are running."""
    if args.fake_servers:
        logger.info("Skipping server checks (Fake servers mode active).")
        return

    try:
        cfg = load_config('config/server_config.yaml')
        failed = []

        # Check VLM Gateway
        vlm = cfg['vlm_service']['gateway']
        if not check_server(vlm['host'], vlm['port']):
            failed.append("Prompt Polisher (VLM)")

        # Check Image Gen (only if rendering)
        if args.gen_images_from_aug:
            img = cfg['image_gen_service']
            if not check_server(img['host'], img['port']):
                failed.append("Image Generation")

        if failed:
            logger.error(f"Required services are down: {', '.join(failed)}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to verify services: {e}")
        sys.exit(1)

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
    csv_files = list(in_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {in_dir}")
        return
    logger.info(f"Found {len(csv_files)} files to process.")

    # 4. Initialize Client Strategy
    if args.fake_servers:
        logger.warning("!!! USING FAKE API CLIENTS - NO REAL NETWORK CALLS !!!")
        ClientClass = MockAPIClient
    else:
        ClientClass = APIClient

    # Initialize Resources
    api_client = ClientClass("config/server_config.yaml", "config/experiment_config.yaml")
    
    pipeline = AugmentationPipeline(api_client)

    # SPLIT SEMAPHORES
    vlm_semaphore = asyncio.Semaphore(args.vlm_concurrency)
    img_semaphore = asyncio.Semaphore(args.img_concurrency)
    
    # 5. Build Tasks (Collect ALL tasks from ALL files first)
    tasks = [] 
    data_managers = []

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

            # Get Users
            users = [u for u in dm.df['character'].unique() if isinstance(u, str)]
            logger.info(f"File: {csv_file.name} | Users: {users}")

            for user in users:
                t_logger = TaskLogger(csv_out_dir, csv_file.name, user)
                
                pipeline_config = {
                    'create': args.create_aug,
                    'render': args.gen_images_from_aug,
                    'oracle': args.oracle,
                    'candidate_count': args.candidate_count,
                    'img_output_dir': csv_out_dir / "images"
                }

                # Add to the global list of tasks
                tasks.append(
                    pipeline.run_single_user(
                        dm, user,
                        vlm_semaphore, img_semaphore,
                        t_logger, pipeline_config
                    )
                )

        # 6. Execute All Tasks Parallel
        if tasks:
            logger.info(f"Starting execution of {len(tasks)} character pipelines with VLM concurrency {args.vlm_concurrency} and IMG concurrency {args.img_concurrency}...")
            await asyncio.gather(*tasks)
        else:
            logger.warning("No tasks were created. Check input CSVs.")

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
