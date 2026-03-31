import logging
import os

def setup_logger():
    os.makedirs("results/logs", exist_ok=True)

    logging.basicConfig(
        filename="results/logs/run.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger()