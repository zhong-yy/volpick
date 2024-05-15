# import pkg_resources
import logging
import os as _os
from pathlib import Path as _Path

cache_root = _Path(_os.getenv("VOLPICK_CACHE_ROOT", _Path(_Path.home(), ".volpick")))

logger = logging.getLogger("volpick")
_ch = logging.StreamHandler()
_ch.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(_ch)
logger.propagate = False
