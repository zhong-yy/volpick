import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
from pathlib import Path
import datetime
import logging
import volpick
from volpick.data import NoiseData, AlaskaDataset
import numpy as np
from volpick.data.utils import (
    plot_waveforms,
    plot_spectrum,
    plot_spectrogram,
    check_waveform,
    check_waveforms_parallel,
)

log_level = logging.INFO
logger = logging.getLogger("Download")
logger.propagate = False
logger.setLevel(log_level)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
noise = NoiseData(root_folder_name="Noise")
file_handler = logging.FileHandler(filename=noise.save_dir / "download.log")
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def prepare_catalog():
    hawaii = volpick.data.HawaiiDataset(
        root_folder_name="hawaii2012to2021",
        cat_file_name="hawaii_catalog2012to2021",
    )
    hw_whole_catalog = hawaii.read(format="csv")
    logger.info(f"Creating a reference table for downloading noise")
    noise.create_noise_table(
        base_catalog=hw_whole_catalog,
        number_stations=250,
        time_difference_limit=12 * 3600,
        number_records_each_station=1000,
        output_filename="hw_noise_reference.csv",
    )
    logger.info(f"Downloading the IRIS inventory (station information)")
    noise.get_inventory()


def download_hawaii_noise():
    catalog_table = pd.read_csv(noise.save_dir / "hw_noise_reference.csv")
    num_processes = 16
    t1 = time.perf_counter()
    logger.info(
        f"Starting downloading data. Save dir: {noise.save_dir}. Number of processes: {num_processes}."
    )
    noise.download_data(
        catalog_table=catalog_table,
        time_window=120,
        sampling_rate=100,
        num_processes=num_processes,
        download_dir=noise.save_dir / "hw_mseed",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def check_noise_examples():
    t1 = time.perf_counter()
    num_processes = 60
    catalog_table = pd.read_csv(noise.save_dir / "hw_mseed_log" / "downloads.csv")
    data_dir = noise.save_dir / "hw_mseed"
    logger.info(
        f"""Plotting noise from Hawaii. Save dir: {noise.save_dir/"hw_mseed_fig"}. Number of processes: {num_processes}."""
    )
    check_waveforms_parallel(
        catalog_table, data_dir, num_processes=num_processes, fts=8
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


if __name__ == "__main__":
    # prepare_catalog()
    # download_hawaii_noise()
    # download_alaska_noise()
    check_noise_examples()
