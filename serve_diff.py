# serve_diff.py

import uvicorn
import yaml
import logging
import sys
import time
from threading import Thread
from pydantic import BaseModel, ValidationError

# Import the app instance from the service module
from src.utils import setup_logging, load_config
# Call setup_logging *before* importing the app
# This ensures this launcher's log config takes precedence
setup_logging(log_file='logs/diff_server.log')
logger = logging.getLogger(__name__)

import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'

# Now, import the app. It will use the logging config we just set up.
from src.image_gen_app import app as image_gen_app


# --- Configuration Model  ---
class ServerConfig(BaseModel):
    host: str
    port: int

# Global flag to signal shutdown
shutdown_requested = False

def run_uvicorn(host: str, port: int):
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
    
    try:
        # --- 2. Load and Validate Config ---
        # Use the utility function to load the config
        full_config = load_config(config_path='config/server_config.yaml')
        raw_server_config = full_config.get('image_gen_service')

        if raw_server_config is None:
            raise ValueError("'image_gen_service' section not found in config/server_config.yaml")

        # Validate host and port. This will raise ValidationError if missing.
        server_config = ServerConfig(**raw_server_config)
        
        logger.info(f"Server config validated: Host={server_config.host}, Port={server_config.port}")

    except ValidationError as e:
        logger.critical(f"--- CONFIGURATION ERROR ---")
        logger.critical(f"FATAL: Missing 'host' or 'port' in 'image_gen_service' section of config/server_config.yaml:")
        logger.critical(f"\n{e}")
        sys.exit("Invalid configuration. Please check the log.") # Use sys.exit
    except FileNotFoundError:
        logger.error(f"Configuration file not found at config/server_config.yaml. Cannot start servers.")
        sys.exit("Config file not found.") # Use sys.exit
    except Exception as e:
        logger.error(f"Failed to load config or start servers: {e}", exc_info=True)
        sys.exit("Server startup failed.") # Use sys.exit


    # --- 3. Start Server Thread ---
    # Use the validated server_config object
    uvicorn_thread = Thread(
        target=run_uvicorn,
        args=(server_config.host, server_config.port),
        daemon=True
    )
    uvicorn_thread.start()

    # Keep main thread alive, listen for shutdown signals
    try:
        while not shutdown_requested:
            time.sleep(1) # Check periodically
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, initiating shutdown...")
        shutdown_requested = True


    # Wait for threads to finish
    logger.info("Waiting for servers to shut down...")
    if uvicorn_thread.is_alive():
        logger.info("Uvicorn runs in a separate process; manual stop (Ctrl+C) might be needed if it doesn't exit.")
        # uvicorn_thread.join(timeout=5) # This join is often not effective

    logger.info("Server launcher finished.")
