import logging
from .consts import OutputDirectory, PlotsDirectory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

if not OutputDirectory.exists():
    OutputDirectory.mkdir()

if not PlotsDirectory.exists():
    PlotsDirectory.mkdir()
