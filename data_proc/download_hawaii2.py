import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
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
hawaii = volpick.data.HawaiiDataset(
    root_folder_name="hawaii2012to2021",
    cat_file_name="hawaii_catalog2012to2021",
)
file_handler = logging.FileHandler(filename=hawaii.save_dir / "download.log")
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

    logger.info(f"Downloading the IRIS inventory (station information)")
    hawaii.get_inventory(client_name="iris")

    summary_file = "/mnt/DATA2/YiyuanZhong/LP_catalog/Hawaii/events_2012to20210930.txt"
    station_archive_file = (
        "/mnt/DATA2/YiyuanZhong/LP_catalog/Hawaii/events_2012to20210930.hypo"
    )

    logger.info(f"Reading and converting the catalog ...")
    hawaii.read_catalog(
        summary_file=summary_file,
        station_archive_file=station_archive_file,
        id_prefix="hawaii2_",
        save_csv=True,
        save_quakeml=False,
        on_screen=False,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(
        f"Finished converting the catalog.\r\nThe catalog is saved to {hawaii._save_quakeml_path} and {hawaii._save_csv_path}.\r\nRuning time {running_time}"
    )


def download():
    t1 = time.perf_counter()

    logger.info(f"Reading the whole catalog ...")
    catalog_table = hawaii.read(format="csv")

    # Selecting those traces with both P and S
    logger.info(f"Selecting the data with both P and S ...")
    metadata = catalog_table[
        (pd.notna(catalog_table["trace_s_arrival_time"]))
        & (pd.notna(catalog_table["trace_p_arrival_time"]))
    ].copy()

    # select the same number of events
    lp_table = metadata[metadata["source_type"] == "lp"].copy()
    rg_table = metadata[metadata["source_type"] != "lp"].copy()

    lp_ids = lp_table.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True, inplace=False
    )["source_id"]
    rg_ids = rg_table.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True, inplace=False
    )["source_id"]

    n_lp_events = len(lp_ids)
    n_rg_events = len(rg_ids)
    assert n_rg_events > n_lp_events

    rand_events_idxs = np.sort(
        np.random.default_rng(seed=50).choice(
            rg_ids, size=n_lp_events * 5, replace=False
        )
    )
    rand_rg_table = rg_table[rg_table["source_id"].isin(rand_events_idxs)].copy()
    new_table = pd.concat([rand_rg_table, lp_table], ignore_index=True).copy()
    print(f"""{len(new_table[new_table["source_type"]=="lp"])} lp traces""")
    print(f"""{len(new_table[new_table["source_type"]!="lp"])} regular traces""")
    # downloading
    num_processes = 16
    logger.info(
        f"Starting downloading data (including LP and VT). Save dir: {hawaii.save_dir}. Number of processes: {num_processes}."
    )
    hawaii.download_data(
        catalog_table=new_table,
        time_before=60,
        time_after=60,
        sampling_rate=100,
        num_processes=num_processes,
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished. Runing time {running_time}")


def to_seisbench_format_same():
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(hawaii.save_dir / "mseed_log" / "downloads.csv")
    lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
    rg_table = catalog_table[catalog_table["source_type"] != "lp"].copy()

    # destination path
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/hawaii2012to2021"
    logger.info(f"Start to convert lp waveforms ...")
    # convert lp waveforms
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=hawaii.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_hw12t21_lp",
    )

    n_lp = len(lp_table)
    n_rg = len(rg_table)
    assert n_rg > n_lp

    logger.info(f"Start to convert rg waveforms ...")
    # convert the same number of regular earthquake waveforms
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
        np.random.default_rng(seed=51).choice(
            rg_ids, size=int(1.18 * n_lp_events), replace=False
        )
    )
    rand_events_rg_table = rg_table[rg_table["source_id"].isin(rand_events_idxs)].copy()
    # select the same number of waveforms
    print(len(rand_events_rg_table))
    print(n_lp)
    rand_waveform_idxs = np.sort(
        np.random.default_rng(seed=100).choice(
            len(rand_events_rg_table), size=n_lp, replace=False
        )
    )
    randomly_selected_rg_table = rand_events_rg_table.iloc[rand_waveform_idxs].copy()
    volpick.data.convert_mseed_to_seisbench(
        randomly_selected_rg_table,
        mseed_dir=hawaii.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_hw12t21_rg",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finish format conversion. Running time: {running_time}")


def plot_waveforms_gaps():
    waveform_table = pd.read_csv(hawaii.save_dir / "mseed_log" / "abnormal_traces.csv")
    print(hawaii.save_dir)
    hawaii.plot_waveforms_with_phases_in_gap(
        num=60,
        waveform_table=waveform_table,
    )


def retry():
    t1 = time.perf_counter()
    logger.info(f"Trying resolving failed downloads")
    n = hawaii.retry_failed_downloads(
        time_before=60, time_after=60, sampling_rate=100, num_processes=1
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finished, Runing time {running_time}")
    logger.info(f"After re-downloading, there are {n} failed downloads to be resolved.")


def plot_waveforms():
    waveform_table = pd.read_csv(hawaii.save_dir / "mseed_log" / "downloads.csv")
    waveform_table_lp = waveform_table[waveform_table["source_type"] == "lp"]
    hawaii.plot_waveforms(
        indices=np.arange(100),
        waveform_table=waveform_table_lp,
        xrangemin=-2,
        xrangemax=2,
        data_dir=hawaii.save_dir / "mseed",
        fig_dir=hawaii.save_dir / "mseed_lp_fig",
    )

    waveform_table_rg = waveform_table[waveform_table["source_type"] != "lp"]
    hawaii.plot_waveforms(
        indices=np.arange(100),
        waveform_table=waveform_table_rg,
        xrangemin=-2,
        xrangemax=2,
        data_dir=hawaii.save_dir / "mseed",
        fig_dir=hawaii.save_dir / "mseed_rg_fig",
    )


if __name__ == "__main__":
    # convert_catalog()
    # download()
    # retry()
    # plot_waveforms()
    # plot_waveforms_gaps()
    to_seisbench_format_same()
