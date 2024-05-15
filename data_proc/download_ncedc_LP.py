import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
import datetime
import logging
import volpick

log_level = logging.INFO
logger = logging.getLogger("Download")
logger.propagate = False
logger.setLevel(log_level)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
ncedc = volpick.data.NCEDCDataset()
file_handler = logging.FileHandler(filename=ncedc.save_dir / "download.log")
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

    logger.info(f"Downloading the inventory (station information)")
    ncedc.get_inventory(client_name="NCEDC")

    summary_file = "/mnt/DATA2/YiyuanZhong/LP_catalog/NCEDC/LP.txt"
    station_archive_file = "/mnt/DATA2/YiyuanZhong/LP_catalog/NCEDC/NCEDC_LP.pha"

    logger.info(f"Reading and converting the catalog ...")
    ncedc.read_catalog(
        summary_file=summary_file,
        station_archive_file=station_archive_file,
        id_prefix="ncedc",
        save_csv=True,
        save_quakeml=True,
        on_screen=False,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(
        f"Finished converting the catalog.\r\nThe catalog is saved to {ncedc._save_quakeml_path} and {ncedc._save_csv_path}.\r\nRuning time {running_time}"
    )


def download():
    t1 = time.perf_counter()

    logger.info(f"Reading the whole catalog ...")
    catalog_table = ncedc.read(format="csv")

    # Selecting those traces with both P and S
    logger.info(f"Selecting the data with both P and S ...")
    metadata = catalog_table[
        (pd.notna(catalog_table["trace_s_arrival_time"]))
        & (pd.notna(catalog_table["trace_p_arrival_time"]))
    ].copy()

    num_processes = 16
    logger.info(
        f"Starting downloading data . Save dir: {ncedc.save_dir}. Number of processes: {num_processes}."
    )
    ncedc.download_data(
        catalog_table=metadata,
        time_before=60,
        time_after=60,
        sampling_rate=100,
        client_name="NCEDC",
        num_processes=num_processes,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def retry():
    t1 = time.perf_counter()
    logger.info(f"Trying resolving failed downloads")
    n = ncedc.retry_failed_downloads(
        time_before=60,
        time_after=60,
        sampling_rate=100,
        client_name="NCEDC",
        num_processes=16,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished, Runing time {running_time}")
    logger.info(f"After re-downloading, there are {n} failed downloads to be resolved.")


def to_seisbench_format():
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(ncedc.save_dir / "mseed_log" / "downloads.csv")
    lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
    # rg_table = catalog_table[catalog_table["source_type"] != "lp"]
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/ncedc"
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=ncedc.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_ncedc_lp",
    )
    # volpick.data.convert_mseed_to_seisbench(
    #     rg_table,
    #     mseed_dir=ncedc.save_dir / "mseed",
    #     dest_dir=dest_dir,
    #     chunk="_rg",
    # )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(running_time)


if __name__ == "__main__":
    # convert_catalog()
    # download()
    # retry()
    to_seisbench_format()
