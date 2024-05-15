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
japan = volpick.data.JapanDataset()
file_handler = logging.FileHandler(filename=japan.save_dir / "download.log")
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def convert_catalog():
    t1 = time.perf_counter()

    station_archive_file_dir = "/mnt/DATA2/YiyuanZhong/LP_catalog/Japan/"

    logger.info(f"Reading and converting the catalog ...")
    japan.read_catalog_multiple_files(
        catalog_dir=station_archive_file_dir, num_processes=62, delete_temp_files=True
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(
        f"Finished converting the catalog.\r\nThe catalog is saved to {japan._save_csv_path}. Runing time is {running_time}"
    )


def download_arrival_time_catalogs():
    with open("./NIED_account", "r") as f:
        username = f.readline().strip()
        password = f.readline().strip()
    print(f"User name: {username}")
    print(f"Password: {password}")
    t1 = time.perf_counter()
    volpick.data.JapanDataset.download_jma_unified_catalog(
        save_dir="/mnt/DATA2/YiyuanZhong/LP_catalog/Japan_catalog",
        username=username,
        password=password,
        # startdate=datetime.datetime(2011, 1, 27),
        # enddate=datetime.datetime(2011, 2, 2),
        startdate=datetime.datetime(2004, 4, 1),
        enddate=datetime.datetime(2023, 6, 28),
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")
    print(f"Elapsed time: {running_time}")


def check_catalog_files():
    t1 = time.perf_counter()
    volpick.data.JapanDataset.check_jma_unified_catalog(
        catalog_dir="/mnt/DATA2/YiyuanZhong/LP_catalog/Japan_catalog",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(f"Elapsed time: {running_time}")


if __name__ == "__main__":
    ## download the original 7-day catalogs
    # download_arrival_time_catalogs()
    # check_catalog_files()

    # convert to readable forms
    convert_catalog()
