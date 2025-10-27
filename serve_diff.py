# serve_models.py

import uvicorn
import yaml
import logging
import subprocess
import sys
import time
from threading import Thread

# Import the app instance from the service module
from src.image_gen_app import app as image_gen_app
from src.utils import setup_logging

# Setup logging
setup_logging(log_file='logs/server_launcher.log')
logger = logging.getLogger(__name__)

# Global flag to signal shutdown
shutdown_requested = False

def run_uvicorn(host, port):
    """ Function to run Uvicorn server """
    logger.info(f"Starting Uvicorn server for Image Generation Service on {host}:{port}...")
    try:
        # Note: reload=False for production/stability
        uvicorn.run(image_gen_app, host=host, port=port, log_config=None) # Use root logger
    except Exception as e:
        logger.error(f"Uvicorn server failed: {e}", exc_info=True)
    finally:
        logger.info("Uvicorn server shut down.")
        global shutdown_requested
        shutdown_requested = True # Signal other threads if Uvicorn stops

if __name__ == "__main__":
    config = {}
    config_path='config/server_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        img_gen_conf = config.get('image_gen_service', {})

        # --- Image Gen Server ---
        img_gen_host = img_gen_conf.get('host', '0.0.0.0')
        img_gen_port = img_gen_conf.get('port', 8000)
        uvicorn_thread = Thread(target=run_uvicorn, args=(img_gen_host, img_gen_port), daemon=True)
        uvicorn_thread.start()

        # Keep main thread alive, listen for shutdown signals
        try:
            while not shutdown_requested:
                time.sleep(1) # Check periodically
        except KeyboardInterrupt:
            logger.info("Ctrl+C received, initiating shutdown...")
            shutdown_requested = True


        # Wait for threads to finish (or signal them if needed)
        logger.info("Waiting for servers to shut down...")
        if uvicorn_thread.is_alive():
            # Uvicorn needs to be stopped externally or via its own signal handling
            # This join might just wait indefinitely if Uvicorn doesn't stop
            logger.info("Uvicorn runs in a separate process/loop; manual stop (Ctrl+C) might be needed if it doesn't exit.")
            # uvicorn_thread.join(timeout=5)


        logger.info("Server launcher finished.")


    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Cannot start servers.")
    except Exception as e:
        logger.error(f"Failed to start servers: {e}", exc_info=True)
