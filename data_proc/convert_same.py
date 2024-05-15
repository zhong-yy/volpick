import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
import datetime
import logging
import volpick
import numpy as np


def to_seisbench_format_rg_same():
    t1 = time.perf_counter()
    alsk = volpick.data.AlaskaDataset("/home/zhongyiyuan/DATA/my_data/Alaska0")
    catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")
    lp_table = catalog_table[catalog_table["source_type"] == "lp"]
    rg_table = catalog_table[catalog_table["source_type"] != "lp"]
    n_lp = len(lp_table)
    n_rg = len(rg_table)
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench2/alaska1"

    subset_idxs = np.sort(
        np.random.default_rng(seed=100).choice(n_rg, size=n_lp, replace=False)
    )
    randomly_selected_rg_table = rg_table.iloc[subset_idxs].copy()
    volpick.data.convert_mseed_to_seisbench(
        randomly_selected_rg_table,
        mseed_dir=alsk.save_dir / "mseed",
        dest_dir=dest_dir,
        chunk="_rg",
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    print(running_time)


if __name__ == "__main__":
    to_seisbench_format_rg_same()
