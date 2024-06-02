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
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
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
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finish format conversion. Running time: {running_time}")


def to_seisbench_format_one_phase():
    catalog_table = pd.read_csv(
        hawaii.save_dir / "mseed_one_phase_log" / "downloads.csv"
    )
    t1 = time.perf_counter()
    phas = ["p", "s"]
    lp_table_dict = {}
    rg_table_dict = {}
    for phid in range(2):
        pha = phas[phid]
        pha2 = phas[1 - phid]

        lp_table_dict[pha] = catalog_table[
            (catalog_table["source_type"] == "lp")
            & (pd.notna(catalog_table[f"trace_{pha}_arrival_time"]))
            & (pd.isna(catalog_table[f"trace_{pha2}_arrival_time"]))
        ].copy()

        rg_table_dict[pha] = catalog_table[
            (catalog_table["source_type"] != "lp")
            & (pd.notna(catalog_table[f"trace_{pha}_arrival_time"]))
            & (pd.isna(catalog_table[f"trace_{pha2}_arrival_time"]))
        ].copy()
        # lens.append(len(lp_table_dict[pha]))
        # lens.append(len(rg_table_dict[pha]))

    num_trace = 6000
    for pha in ["p", "s"]:
        lp_table_dict[pha] = lp_table_dict[pha].iloc[
            np.sort(
                np.random.default_rng(seed=102).choice(
                    len(lp_table_dict[pha]), size=num_trace, replace=False
                )
            )
        ]
        rg_table_dict[pha] = rg_table_dict[pha].iloc[
            np.sort(
                np.random.default_rng(seed=98).choice(
                    len(rg_table_dict[pha]), size=num_trace, replace=False
                )
            )
        ]
    lp_table = pd.concat(
        [lp_table_dict["p"], lp_table_dict["s"]], ignore_index=True
    ).copy()

    rg_table = pd.concat(
        [rg_table_dict["p"], rg_table_dict["s"]], ignore_index=True
    ).copy()

    # destination path
    dest_dir = "/home/zhongyiyuan/DATA/my_datasets_seisbench/hawaii_onephase"
    logger.info(f"Start to convert lp waveforms p or s ...")

    # convert lp waveforms
    volpick.data.convert_mseed_to_seisbench(
        lp_table,
        mseed_dir=hawaii.save_dir / "mseed_one_phase",
        dest_dir=dest_dir,
        chunk=f"_hw12t21_onephase_lp",
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
    )

    volpick.data.convert_mseed_to_seisbench(
        rg_table,
        mseed_dir=hawaii.save_dir / "mseed_one_phase",
        dest_dir=dest_dir,
        chunk=f"_hw12t21_onephase_rg",
        skip_spikes=True,
        split_prob=[0.85, 0.05, 0.1],
    )
    t2 = time.perf_counter()
    running_time = str(datetime.timedelta(seconds=t2 - t1))
    logger.info(f"Finish format conversion. Running time: {running_time}")


if __name__ == "__main__":
    # to_seisbench_format_same()
    to_seisbench_format_one_phase()
