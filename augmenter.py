# augmenter.py

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.utils import setup_logging, check_server, load_config
from src.data_manager import ConversationDataManager
from src.pipeline import AugmentationPipeline
from src.api_clients import APIClient
from src.logger import TaskLogger

# Root logger for the orchestrator script (prints to stdout)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ORCHESTRATOR] - %(message)s')
logger = logging.getLogger("Orchestrator")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parallel Augmentation Orchestrator")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing CSVs (e.g. data/exp1)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory (e.g. output/exp1)")
    
    parser.add_argument("--create_aug", action='store_true', help="Run Stage 1: Generate prompts")
    parser.add_argument("--gen_images_from_aug", action='store_true', help="Run Stage 2: Render images")
    
    parser.add_argument("--concurrency", type=int, default=8, help="Global Max Concurrent API Requests")
    parser.add_argument("--oracle", action='store_true', help="Oracle Context Mode")
    parser.add_argument("--mock", action='store_true', help="Mock Mode")
    
    return parser.parse_args()

def verify_services(args):
    """Checks if required servers are running (unless in mock mode)."""
    if args.mock:
        logger.info("Running in MOCK MODE. Skipping server checks.")
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
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    
    # 1. Setup Directory Structure
    # Output structure: output_dir / [logs, traces, images, data]
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        logger.error(f"Input directory does not exist: {in_dir}")
        sys.exit(1)

    # 2. Verify Services (unless mock)
    verify_services(args)


    # 3. Discovery
    csv_files = list(in_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {in_dir}")
        return

    logger.info(f"Found {len(csv_files)} files to process.")

    # 4. Initialize Shared Resources
    api_client = APIClient("config/server_config.yaml", "config/experiment_config.yaml")
    pipeline = AugmentationPipeline(api_client, mock_mode=args.mock)
    
    # GLOBAL SEMAPHORE: Controls total active API requests across ALL files/users
    global_sem = asyncio.Semaphore(args.concurrency)
    
    tasks = []
    
    # 5. Build Tasks
    # We hold references to data managers to save them at the very end as well
    data_managers = []

    async with api_client:
        for csv_file in csv_files:
            # Output CSV path (combined for all users in this file)
            aug_csv_path = out_dir / "data" / f"{csv_file.stem}_augmented.csv"
            
            # Create DataManager (Shared per file)
            dm = ConversationDataManager(str(csv_file), str(aug_csv_path))
            dm.load_and_prepare()
            dm.save_lock = asyncio.Lock() # Ensure safety
            data_managers.append(dm)

            # Get Users
            users = [u for u in dm.df['character'].unique() if isinstance(u, str)]
            logger.info(f"File: {csv_file.name} | Users: {users}")

            # Create a Task for each User
            for user in users:
                # Each user gets their own dedicated Logger
                t_logger = TaskLogger(out_dir, csv_file.name, user)
                
                # Config payload for the pipeline
                pipeline_config = {
                    'create': args.create_aug,
                    'render': args.gen_images_from_aug,
                    'oracle': args.oracle,
                    'img_output_dir': out_dir / "images"
                }

                # Schedule the task
                tasks.append(
                    pipeline.run_single_user(dm, user, global_sem, t_logger, pipeline_config)
                )

        # 6. Execute All Tasks Parallel
        logger.info(f"Starting execution of {len(tasks)} character pipelines with concurrency {args.concurrency}...")
        await asyncio.gather(*tasks)

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
