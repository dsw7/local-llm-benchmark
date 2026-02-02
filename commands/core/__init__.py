import logging
from .consts import DIR_OUTPUT

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if not DIR_OUTPUT.exists():
    DIR_OUTPUT.mkdir()
