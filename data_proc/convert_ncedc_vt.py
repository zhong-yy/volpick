import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import numpy as np
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
ncedc = volpick.data.NCEDCDataset(
    root_folder_name="ncedc_vt", cat_file_name="ncedc_catalog_vt", etype="e"
)
file_handler = logging.FileHandler(filename=ncedc.save_dir / "download.log")
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def to_seisbench_format():
    t1 = time.perf_counter()
    catalog_table = pd.read_csv(
        volpick.cache_root / "ncedc_vt" / "mseed_log" / "downloads.csv"
    )
    vt_table = catalog_table[catalog_table["source_type"] != "lp"].copy()

    eids = vt_table.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True, inplace=False
    )["source_id"]
    print(len(vt_table))
    rand_events_idxs = np.sort(
        np.random.default_rng(seed=90).choice(eids, size=1900, replace=False)
    )

    rand_events_table = vt_table[vt_table["source_id"].isin(rand_events_idxs)].copy()
    print(len(rand_events_table))

    rand_waveform_idxs = np.sort(
        np.random.default_rng(seed=100).choice(
            len(rand_events_table), size=6900, replace=False
        )
    )
    randomly_selected_traces = rand_events_table.iloc[rand_waveform_idxs].copy()

    # rg_table = catalog_table[catalog_table["source_type"] != "lp"]
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/ncedc"
    volpick.data.convert_mseed_to_seisbench(
        randomly_selected_traces,
        mseed_dir=volpick.cache_root / "ncedc_vt" / "mseed",
        dest_dir=dest_dir,
        chunk="_ncedc_vt",
        skip_spikes=True,
        # split_prob=[0.85, 0.05, 0.1],
        split_prob=[0, 0, 1],
        n_limit=4841,
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
