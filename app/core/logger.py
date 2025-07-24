import logging

logger = logging.getLogger("geo-analyzer")
logger.setLevel(logging.INFO)  # Change to DEBUG for more detail

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
