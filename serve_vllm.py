# serve_vllm.py

import yaml
import logging
import subprocess
import sys
import time
from threading import Thread

# Import the app instance from the service module
from src.utils import setup_logging, load_config

# Setup logging
setup_logging(log_file='logs/server_launcher.log')
logger = logging.getLogger(__name__)

config = load_config('config/server_config.yaml')
config = config.get("prompt_enhancer_service")

# Global flag to signal shutdown
shutdown_requested = False

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
        "--max_model_len", str(25000),
        "--seed", str(config.get("seed")),
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
    try:
        # --- Prompt Enhancer Server ---
        enhancer_host = config.get('host', '0.0.0.0')
        enhancer_port = config.get('port', 8001)
        enhancer_model = config.get('model_id')
        tp = config.get('tensor_parallel_size', 1)
        mem_util = config.get('gpu_memory_utilization', 0.25)


        vllm_thread = Thread(target=run_vllm, args=(enhancer_host, enhancer_port, enhancer_model, tp, mem_util), daemon=True)
        vllm_thread.start()

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

        logger.info("Server launcher finished.")


    except FileNotFoundError:
        logger.error(f"Configuration file not found. Cannot start servers.")
    except Exception as e:
        logger.error(f"Failed to start servers: {e}", exc_info=True)
