import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
import datetime
import logging
import volpick
import numpy as np


def to_seisbench_format_vol_lp():
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "Vol_lp")
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/japan_vol_lp"
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=japan.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_japan_vol_lp",
        is_japan_format=True,
        check_long_traces=True,
        check_long_traces_limit=120,
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(running_time)


def to_seisbench_format_tec_lp():
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "Tec_lp")
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    lp_table = catalog_table[catalog_table["source_type"] == "lp"].copy()
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/japan_tec_lp"
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=japan.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_japan_tec_lp",
        is_japan_format=True,
        check_long_traces=True,
        check_long_traces_limit=120,
        skip_spikes=True,
        # split_prob=[0.85, 0.05, 0.1],
        split_prob=[0, 0, 1],
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(running_time)


def to_seisbench_format_vt():
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "VT")
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    vt_table = catalog_table[catalog_table["source_type"] != "lp"].iloc[0:82650].copy()
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/japan_vol_lp"
    volpick.data.convert_mseed_to_seisbench(
        vt_table,
        mseed_dir=japan.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_japan_vol_lp",
        is_japan_format=True,
        check_long_traces=True,
        check_long_traces_limit=120,
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(running_time)


if __name__ == "__main__":
    to_seisbench_format_vol_lp()
