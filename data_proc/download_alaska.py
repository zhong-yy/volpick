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
alsk = volpick.data.AlaskaDataset()
file_handler = logging.FileHandler(filename=alsk.save_dir / "download.log")
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
    alsk = volpick.data.AlaskaDataset()

    logger.info(f"Downloading the IRIS inventory (station information)")
    alsk.get_inventory()

    summary_file = "/mnt/DATA2/YiyuanZhong/LP_catalog/Alaska/Power_SIR-5037_Catalog_Data/summary/avo_1989-2018_hypoi.summary"
    station_archive_file = "/mnt/DATA2/YiyuanZhong/LP_catalog/Alaska/Power_SIR-5037_Catalog_Data/summary/avo_1989-2018_hypoi.pha"

    logger.info(f"Reading and converting the catalog ...")
    alsk.read_catalog(
        summary_file=summary_file,
        station_archive_file=station_archive_file,
        id_prefix="alk",
        save_csv=True,
        save_quakeml=False,
        on_screen=True,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(
        f"Finished converting the catalog.\r\nThe catalog is saved to {alsk._save_quakeml_path} and {alsk._save_csv_path}.\r\nRuning time {running_time}"
    )


def download_lp():
    alsk = volpick.data.AlaskaDataset()
    logger.info(f"Starting downloading lp data ...")
    t1 = time.perf_counter()

    logger.info(f"Reading the catalog ...")
    catalog_table = alsk.read(format="csv")

    # Select lp
    logger.info(f"Selecting only LP data ...")
    lp_metadata = catalog_table[catalog_table["source_type"] == "lp"].copy()

    # Selecting those traces with both P and S
    logger.info(f"Selecting the data with both P and S ...")
    lp_metadata = lp_metadata[
        (pd.notna(lp_metadata["trace_s_arrival_time"]))
        & (pd.notna(lp_metadata["trace_p_arrival_time"]))
    ].copy()

    alsk = volpick.data.AlaskaDataset()
    alsk.download_data(
        catalog_table=lp_metadata,
        time_before=60,
        time_after=60,
        sampling_rate=100,
        num_processes=16,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def retry_lp():
    alsk = volpick.data.AlaskaDataset()
    logger.info(f"Trying resolving failed downloads")
    t1 = time.perf_counter()
    alsk.retry_failed_downloads(
        time_before=60, time_after=60, sampling_rate=100, num_processes=6
    )

    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished, Runing time {running_time}")


def download():
    alsk = volpick.data.AlaskaDataset()
    t1 = time.perf_counter()

    logger.info(f"Reading the whole catalog ...")
    catalog_table = alsk.read(format="csv")

    # Selecting those traces with both P and S
    logger.info(f"Selecting the data with both P and S ...")
    metadata = catalog_table[
        (pd.notna(catalog_table["trace_s_arrival_time"]))
        & (pd.notna(catalog_table["trace_p_arrival_time"]))
    ].copy()

    alsk = volpick.data.AlaskaDataset()
    num_processes = 16
    logger.info(
        f"Starting downloading data (including LP and VT). Save dir: {alsk.save_dir}. Number of processes: {num_processes}."
    )
    alsk.download_data(
        catalog_table=metadata,
        time_before=60,
        time_after=60,
        sampling_rate=100,
        num_processes=num_processes,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def retry():
    t1 = time.perf_counter()
    alsk = volpick.data.AlaskaDataset()
    logger.info(f"Trying resolving failed downloads")

    n = alsk.retry_failed_downloads(
        time_before=60, time_after=60, sampling_rate=100, num_processes=16
    )
    logger.info(f"After re-downloading, there are {n} failed downloads to be resolved.")
    while n != 0:
        n = alsk.retry_failed_downloads(
            time_before=60, time_after=60, sampling_rate=100, num_processes=16
        )
        logger.info(
            f"After re-downloading, there are {n} failed downloads to be resolved."
        )

    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished, Runing time {running_time}")


# def to_seisbench_format():
#     t1 = time.perf_counter()
#     alsk = volpick.data.AlaskaDataset()
#     catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")
#     lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
#     rg_table = catalog_table[catalog_table["source_type"] != "lp"].copy()
#     dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench2/alaska"
#     volpick.data.convert_mseed_to_seisbench(
#         lp_table,
#         mseed_dir=alsk.save_dir / "mseed",
#         dest_dir=dest_dir,
#         chunk="_lp",
#     )
#     volpick.data.convert_mseed_to_seisbench(
#         rg_table,
#         mseed_dir=alsk.save_dir / "mseed",
#         dest_dir=dest_dir,
#         chunk="_rg",
#     )
#     t2 = time.perf_counter()
#     running_time = str(datetime.timedelta(seconds=t2 - t1))
#     print(running_time)


def to_seisbench_format_same():
    t1 = time.perf_counter()
    # alsk = volpick.data.AlaskaDataset()

    catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")

    lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
    rg_table = catalog_table[catalog_table["source_type"] != "lp"].copy()

    # destination path
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/alaska"
    logger.info(f"Start to convert lp waveforms ...")
    # convert lp waveforms
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=alsk.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_ak_lp",
    )

    # Convert the same number of regular earthquake waveform traces
    n_lp = len(lp_table)
    n_rg = len(rg_table)
    assert n_rg > n_lp

    logger.info(f"Start to convert rg waveforms ...")

    lp_ids = lp_table.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True, inplace=False
    )["source_id"]
    rg_ids = rg_table.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True, inplace=False
    )["source_id"]
    n_lp_events = len(lp_ids)
    n_rg_events = len(rg_ids)
    assert n_rg_events > n_lp_events

    # select the same number of events
    rand_events_idxs = np.sort(
        np.random.default_rng(seed=50).choice(rg_ids, size=n_lp_events, replace=False)
    )
    rand_events_rg_table = rg_table[rg_table["source_id"].isin(rand_events_idxs)].copy()
    # select the same number of waveforms
    rand_waveform_idxs = np.sort(
        np.random.default_rng(seed=100).choice(
            len(rand_events_rg_table), size=n_lp, replace=False
        )
    )
    randomly_selected_rg_table = rand_events_rg_table.iloc[rand_waveform_idxs].copy()
    volpick.data.convert_mseed_to_seisbench(
        randomly_selected_rg_table,
        mseed_dir=alsk.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_ak_rg",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finish format conversion. Running time: {running_time}")


if __name__ == "__main__":
    # convert_catalog()
    download()
    retry()
    # to_seisbench_format()
    to_seisbench_format_same()
