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
        skip_spikes=True,
        # split_prob=[0.85, 0.05, 0.1],
        split_prob=[0, 0, 1],
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
