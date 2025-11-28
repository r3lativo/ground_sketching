# augmenter.py

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

from src.utils import setup_logging, check_server, load_config
from src.data_manager import ConversationDataManager
from src.pipeline import AugmentationPipeline
from src.api_clients import APIClient

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ground Sketching Augmentation Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--user", type=str, help="Specific character perspective to process")
    parser.add_argument("--automatic_users", action='store_true', help="Automatically process all characters found in the file")
    
    parser.add_argument("--create_aug", action='store_true', help="Run Stage 1: Generate/Refine text prompts")
    parser.add_argument("--gen_images_from_aug", action='store_true', help="Run Stage 2: Render images from prompts")
    
    parser.add_argument("--aug_output_path", type=str, default="output/", help="Directory or file path for the augmented CSV")
    parser.add_argument("--images_output_path", type=str, default="output/", help="Directory for generated images")
    
    parser.add_argument("--realistic_chunk", action='store_true', help="Limit context strictly to previous utterances (no future context)")
    parser.add_argument("--mock", action='store_true', help="Run in mock mode (No GPU/API required)")
    
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

def configure_paths(args):
    """Configures input/output paths based on arguments and mock mode."""
    input_path = Path(args.file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.file}")
        sys.exit(1)

    # Mock Mode Overrides
    if args.mock:
        mock_root = Path("output/mock")
        aug_out = mock_root / f"{input_path.stem}_aug.csv"
        img_out = mock_root / f"{input_path.stem}_images"
        
        # Cleanup mock data for fresh run behavior
        if aug_out.exists(): aug_out.unlink()
        if img_out.exists(): shutil.rmtree(img_out)
        
        return input_path, aug_out, img_out

    # Standard Mode
    if args.aug_output_path == "output/":
        aug_out = Path("output") / f"{input_path.stem}_aug.csv"
    else:
        aug_out = Path(args.aug_output_path)

    if args.images_output_path == "output/":
        img_out = Path("output") / f"{input_path.stem}_images"
    else:
        img_out = Path(args.images_output_path)

    # Make sure folders exist
    aug_out.parent.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    return input_path, aug_out, img_out

async def main():
    setup_logging()
    args = parse_arguments()
    
    # 1. Setup Environment
    input_path, aug_output_path, img_output_dir = configure_paths(args)
    verify_services(args)

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {aug_output_path}")
    logger.info(f"Images: {img_output_dir}")

    # 2. Initialize Data Layer
    data_manager = ConversationDataManager(str(input_path), str(aug_output_path))
    data_manager.load_and_prepare()

    # 3. Initialize Pipeline
    api_client = APIClient("config/server_config.yaml", "config/experiment_config.yaml")
    pipeline = AugmentationPipeline(data_manager, mock_mode=args.mock, api_client=api_client)

    # 4. Determine Users to Process
    users = []
    if args.automatic_users:
        users = [u for u in data_manager.df['character'].unique() if isinstance(u, str)]
    elif args.user:
        users = [args.user]
    
    if not users:
        logger.warning("No users selected to process. Use --user or --automatic_users.")
        return

    # 5. Run Full Pipeline (Parallel Users & Pipelined Stages)
    # Helper lambda to resolve chunk column per user
    def chunk_resolver(u):
        col = f"chunk_{u}"
        return col if col in data_manager.df.columns else "chunk_id"

    await pipeline.run_full_pipeline(
        users=users,
        create=args.create_aug,
        render=args.gen_images_from_aug,
        realistic_context=args.realistic_chunk,
        output_dir=str(img_output_dir),
        chunk_col_resolver=chunk_resolver
    )

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    except Exception as e:
        logger.error(f"Critical failure: {e}", exc_info=True)
        sys.exit(1)
