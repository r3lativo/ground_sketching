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

def run_vllm(host, port, model_id, tp_size=1, gpu_mem_util=0.3):
    """ Function to run vLLM OpenAI-compatible server """
    global shutdown_requested
    logger.info(f"Starting vLLM server for Prompt Enhancer Service on {host}:{port}...")
    logger.info(f"Using model: {model_id}")

    command = [
        sys.executable,  # Use the current Python interpreter
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]

    logger.info(f"Executing command: {' '.join(command)}")
    process = None
    try:
        # Redirect stdout/stderr maybe to separate logs later if needed
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Log vLLM output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if shutdown_requested: # Check if shutdown was requested
                    logger.warning("Shutdown requested, terminating vLLM server...")
                    process.terminate()
                    try:
                        process.wait(timeout=10) # Wait a bit for graceful shutdown
                    except subprocess.TimeoutExpired:
                        logger.warning("vLLM did not terminate gracefully, killing.")
                        process.kill()
                    break
                logger.info(f"[vLLM] {line.strip()}")

        process.wait() # Wait for process to finish if loop breaks normally
        logger.info(f"vLLM server process exited with code: {process.returncode}")

    except FileNotFoundError:
        logger.error("Error: 'python -m vllm...' command failed. Is vLLM installed correctly?")
    except Exception as e:
        logger.error(f"vLLM server failed: {e}", exc_info=True)
        if process:
            process.kill() # Ensure it's killed on unexpected error
    finally:
        logger.info("vLLM server shut down.")
        shutdown_requested = True # Signal Uvicorn if vLLM stops unexpectedly


if __name__ == "__main__":
    config = {}
    config_path='config/server_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        img_gen_conf = config.get('image_gen_service', {})
        enhancer_conf = config.get('prompt_enhancer_service', {})

        # --- Image Gen Server ---
        img_gen_host = img_gen_conf.get('host', '0.0.0.0')
        img_gen_port = img_gen_conf.get('port', 8000)
        uvicorn_thread = Thread(target=run_uvicorn, args=(img_gen_host, img_gen_port), daemon=True)
        uvicorn_thread.start()

        # --- Prompt Enhancer Server ---
        enhancer_host = enhancer_conf.get('host', '0.0.0.0')
        enhancer_port = enhancer_conf.get('port', 8001)
        enhancer_model = enhancer_conf.get('model_id')
        tp = enhancer_conf.get('tensor_parallel_size', 1)
        mem_util = enhancer_conf.get('gpu_memory_utilization', 0.25)

        if enhancer_model: # Only start if model_id is configured
            vllm_thread = Thread(target=run_vllm, args=(enhancer_host, enhancer_port, enhancer_model, tp, mem_util), daemon=True)
            vllm_thread.start()
        else:
            logger.warning("Prompt enhancer model_id not found in config. Skipping vLLM server launch.")
            vllm_thread = None

        # Keep main thread alive, listen for shutdown signals
        try:
            while not shutdown_requested:
                time.sleep(1) # Check periodically
        except KeyboardInterrupt:
            logger.info("Ctrl+C received, initiating shutdown...")
            shutdown_requested = True


        # Wait for threads to finish (or signal them if needed)
        logger.info("Waiting for servers to shut down...")
        if vllm_thread and vllm_thread.is_alive():
            # vLLM thread should stop itself when shutdown_requested is True
             vllm_thread.join(timeout=15) # Wait with timeout
             if vllm_thread.is_alive():
                 logger.warning("vLLM thread did not exit cleanly.")
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
