"""
Structured logger — human-readable console + JSON file
========================================================
Every log call emits:
  1. Formatted line to stdout (existing behaviour)
  2. JSON record to logs/crawl.jsonl (new)

JSON record format:
  {ts, level, logger, worker_id, event, message}
"""

import logging
import sys
import json
import os
import time
import threading

LOG_FILE   = "logs/crawl.jsonl"
_file_lock = threading.Lock()


class JSONFileHandler(logging.Handler):
    """Writes structured JSON records to a .jsonl file."""

    def __init__(self, filepath):
        super().__init__()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath

    def emit(self, record):
        try:
            import re
            from datetime import datetime
            event = {
                "ts":      time.time(),
                "iso":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                "level":   record.levelname,
                "logger":  record.name,
                "message": record.getMessage(),
            }
            msg = record.getMessage()
            m = re.search(r"rank\s+(\d+)", msg)
            if m:
                event["worker_rank"] = int(m.group(1))
            m = re.search(r"https?://\S+", msg)
            if m:
                event["url"] = m.group(0)

            with _file_lock:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
        except Exception:
            pass

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # ── console handler (human readable) ─────────────────────────────
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s -- %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(console)

        # ── JSON file handler (structured) ────────────────────────────────
        json_handler = JSONFileHandler(LOG_FILE)
        json_handler.setLevel(logging.INFO)
        logger.addHandler(json_handler)

    logger.setLevel(logging.INFO)
    return logger