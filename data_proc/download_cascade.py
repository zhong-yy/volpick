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
pnsn = volpick.data.ComCatDataset(
    root_folder_name="Cascade_lp", catfile="cascade_catalog_lp"
)
file_handler = logging.FileHandler(filename=pnsn.save_dir / "download.log")
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def download_phases():
    logger.info(f"Reading lp events downloaded from PNSN ...")
    # How to download lfe events from PSN: https://pnsn.org/events?mag_min=-2&mag_max=10&date_start=1969-2-14&date_end=2024-5-3&lat_min=&lat_max=&lon_min=&lon_max=&city_center=&lat_center=&lon_center=&radius=&depth_min=-5&depth_max=1000&etypes%5B%5D=lf&gap_max=&distance=&phase_min=&s_phase_min=&rms_max=&sort_by=event_time_utc&order=desc
    pnsn = volpick.data.ComCatDataset(
        root_folder_name="Cascade_lp", cat_file_name="cascade_catalog_lp"
    )
    logger.info(f"Downloading phases from ANSS ComCat ...")
    summary_df = pnsn.read_PNSN_events(
        "/home/zhongyiyuan/DATA/LP_catalog/Cascade/LP.csv", source_type="lp"
    )
    pnsn.download_phases(summary_df)
    logger.info(f"Finished downloading phases from ANSS ComCat ...")


def download():
    t1 = time.perf_counter()
    pnsn = volpick.data.ComCatDataset(
        root_folder_name="Cascade_lp", cat_file_name="cascade_catalog_lp"
    )
    logger.info(f"Reading the whole catalog ...")
    catalog_table = pnsn.read(format="csv")

    # Selecting those traces with both P and S
    logger.info(f"Selecting the data with both P and S ...")
    metadata = catalog_table[
        (pd.notna(catalog_table["trace_s_arrival_time"]))
        & (pd.notna(catalog_table["trace_p_arrival_time"]))
    ].copy()

    num_processes = 4
    logger.info(
        f"Starting downloading data . Save dir: {pnsn.save_dir}. Number of processes: {num_processes}."
    )
    pnsn.download_data(
        catalog_table=metadata,
        time_before=60,
        time_after=60,
        sampling_rate=100,
        num_processes=num_processes,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


if __name__ == "__main__":
    download_phases()
    download()
