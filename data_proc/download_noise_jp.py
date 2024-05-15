import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
import datetime
import logging
import volpick
import numpy as np
from volpick.data.utils import (
    plot_waveforms,
    plot_spectrum,
    plot_spectrogram,
    check_waveform,
    check_waveforms_parallel,
)
from volpick.data import NoiseData, JapanDataset, JapanNoiseData

log_level = logging.INFO
logger = logging.getLogger("Download")
logger.propagate = False
logger.setLevel(log_level)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
)
jp_noise = JapanNoiseData(root_folder_name="Noise")
file_handler = logging.FileHandler(filename=jp_noise.save_dir / "download.log")
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def prepare_catalog():
    jp_cat = pd.read_csv(volpick.cache_root / "Japan" / "japan_catalog.csv")

    logger.info(f"Creating a reference table for downloading noise from Japan")
    jp_noise.create_noise_table(
        base_catalog=jp_cat,
        time_difference_limit=2400,
        output_filename="japan_noise_reference_time.csv",
    )


def download_japan_noise():
    reference_table = pd.read_csv(jp_noise.save_dir / "japan_noise_reference_time.csv")
    japan_vol_LP_with_PS = pd.read_csv(
        volpick.cache_root / "Japan" / "japan_vol_LP_with_PS.csv"
    )
    japan_VT_with_PS = pd.read_csv(
        volpick.cache_root / "Japan" / "japan_VT_with_PS.csv"
    )
    t1 = time.perf_counter()
    logger.info(
        f"Starting downloading noise waveforms from japan. Save dir: {jp_noise.save_dir}."
    )
    stations = list(
        np.unique(
            np.append(
                np.unique(japan_vol_LP_with_PS["station_code"].values),
                np.unique(japan_VT_with_PS["station_code"].values),
            )
        )
    )
    jp_noise.download(
        catalog_table=reference_table,
        stations=stations,
        username="zyy123",
        password="zhong95",
        sampling_rate=100,
        download_dir=jp_noise.save_dir / "jp_mseed2",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def check_noise_examples():
    t1 = time.perf_counter()
    num_processes = 55
    catalog_table = pd.read_csv(jp_noise.save_dir / "jp_mseed2_log" / "downloads.csv")
    data_dir = jp_noise.save_dir / "jp_mseed2"
    logger.info(
        f"""Plotting noise from Japan. Save dir: {jp_noise.save_dir/"jp_mseed2_fig"}. Number of processes: {num_processes}."""
    )
    check_waveforms_parallel(
        catalog_table, data_dir, num_processes=num_processes, fts=8, skip_threshold=0.03
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(f"Running time: {running_time}")
    # logger.info(f"Finished. Runing time {running_time}")


if __name__ == "__main__":
    # prepare_catalog()
    # download_japan_noise()
    check_noise_examples()
