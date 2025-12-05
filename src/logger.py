import logging
import json
import os
from pathlib import Path

class TaskLogger:
    """
    Sets up a separate log file for a specific task (User + File)
    and manages the JSONL trace file for model thoughts.
    """
    def __init__(self, output_dir: Path, filename: str, character: str):
        self.char_sanitized = "".join([c for c in character if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        self.filename_stem = Path(filename).stem
        self.task_id = f"{self.filename_stem}_{self.char_sanitized}"
        
        # Paths
        self.log_dir = output_dir / "logs"
        self.trace_dir = output_dir / "traces"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python Logger
        self.logger = logging.getLogger(self.task_id)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Don't bubble up to root logger (avoid scrambled stdout)

        # File Handler (logs/file_char.log)
        fh = logging.FileHandler(self.log_dir / f"{self.task_id}.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)

        # Trace File Path
        self.trace_path = self.trace_dir / f"{self.task_id}_trace.jsonl"
        # Clear previous trace if exists
        if self.trace_path.exists():
            self.trace_path.unlink()

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_trace(self, index: int, step_type: str, raw_response: dict):
        """
        Saves the raw model thought/decision to a JSONL file.
        """
        entry = {
            "index": index,
            "step": step_type,
            "character": self.char_sanitized,
            "file": self.filename_stem,
            "response": raw_response
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
