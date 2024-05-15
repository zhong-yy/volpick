import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
import datetime
import logging
import volpick
import numpy as np


log_level = logging.INFO
logger = logging.getLogger("Download")
logger.propagate = False
logger.setLevel(log_level)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
# japan = volpick.data.JapanDataset()
file_handler = logging.FileHandler(
    filename=volpick.cache_root / "Japan" / "download.log"
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def download_japan_vol_lp_event_waveforms():
    with open("./NIED_account", "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()
    print(f"User name: {username}")
    print(f"Password: {password}")

    t1 = time.perf_counter()
    logger.info(f"Downloading volcanic lp waveforms ...")

    catalog_table = pd.read_csv(
        volpick.cache_root / "Japan" / "japan_vol_LP_with_PS.csv"
    )
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "Vol_lp")

    japan._download(
        catalog_table=catalog_table,
        username=username,
        password=password,
        sampling_rate=100,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(f"Elapsed time: {running_time}")


if __name__ == "__main__":
    download_japan_vol_lp_event_waveforms()
