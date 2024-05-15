from collections import defaultdict
import logging
from pathlib import Path
from abc import abstractmethod, ABC
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy import read
from obspy import Stream
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth

from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNTimeoutException,
    FDSNNoServiceException,
    FDSNServiceUnavailableException,
)

from obspy.clients.fdsn.client import FDSNException
from http.client import HTTPException
from http.client import IncompleteRead
from socket import timeout as socket_timeout

import seisbench
import seisbench.data as sbd
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    stream_to_array,
    trace_has_spikes,
    waveform_id_to_network_station_location,
)

import HinetPy
from HinetPy.utils import to_datetime, point_inside_circular, point_inside_box

import obspy.core.event as obe

from seisbench.data.ethz import InventoryMapper

import volpick

from datetime import datetime
from datetime import timedelta

import shutil


def _read_sac_info(fname):
    with open(fname, "r") as f:
        line = f.readline()
        info_dict = {}
        while line:
            key, _, value = line.partition(":")
            info_dict[key] = value.strip().split()
            line = f.readline()
    return info_dict


def _read_sac_files(data_dir, t_offset):
    sac_files = list(data_dir.glob("*.sac"))
    info_files = [Path(str(x).replace("sac", "pick")) for x in sac_files]
    sts = Stream()
    for sac, info in zip(sac_files, info_files):
        st = read(sac)
        info_dict = _read_sac_info(info)
        start_t = info_dict["start_time"]
        st[0].stats.starttime = UTCDateTime(
            year=int(start_t[0].strip()),
            month=int(start_t[1].strip()),
            day=int(start_t[2].strip()),
            hour=int(start_t[3].strip()),
            minute=int(start_t[4].strip()),
            second=00,
        ) + float(start_t[5])
        for tr in st:
            tr.stats.starttime = tr.stats.starttime + t_offset
        sts += st
    return sts


def check_V_Z(
    catalog_table,
    src_dir=None,
    dest_dir=None,
    log_level=logging.INFO,
):
    # If the process is the main process (sequential downloading),
    # first check and create download_dir and log_dir
    if mp.parent_process() is None:  # check whether it is the main process
        if isinstance(dest_dir, str):
            dest_dir = Path(dest_dir)
        try:
            dest_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{dest_dir} exists")

    # log_dir = self.save_dir / "sac2mseed_log"
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    log_dir = dest_dir / "log"
    if mp.parent_process() is None:  # check whether it is the main process
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{log_dir} exists")

    if mp.parent_process() is None:
        process_mark = ""
        print("It is the main process")
    else:
        process_mark = mp.current_process().name

    # Initialize a logger
    logger = volpick.logger.getChild("convert" + process_mark)
    logger.setLevel(log_level)
    last_subdir = None
    sts = Stream()

    catalog_table.drop_duplicates(
        subset=["source_id", "station_code"], keep="first", inplace=True
    )

    data_with_VZ = []
    abnormal_traces = []
    for row in catalog_table.itertuples(name="trace"):
        event_params = {
            "source_id": row.source_id,
            "source_origin_time": row.source_origin_time,
            "source_latitude_deg": row.source_latitude_deg,
            "source_longitude_deg": row.source_longitude_deg,
            "source_depth_km": row.source_depth_km,
            "source_magnitude": row.source_magnitude,
            "source_magnitude_type": row.source_magnitude_type,
            "source_type": row.source_type,
        }
        trace_params = {
            "station_network_code": row.station_network_code,
            "station_code": row.station_code,
            "station_location_code": row.station_location_code,
            "trace_channel": row.trace_channel,
            "trace_p_arrival_time": row.trace_p_arrival_time,
            "trace_s_arrival_time": row.trace_s_arrival_time,
            "trace_p_max_weight": row.trace_p_max_weight,
            "trace_s_max_weight": row.trace_s_max_weight,
            "trace_p_first_motion": row.trace_p_first_motion,
        }
        waveform_information = (
            f"""{event_params["source_id"]}: {event_params["source_origin_time"]} | """
            f"{trace_params['station_network_code']}.{trace_params['station_code']}"
            f".{trace_params['station_location_code']}.{trace_params['trace_channel']}* |"
        )
        net = row.station_network_code
        sta = row.station_code
        year = event_params["source_origin_time"].split("T")[0].split("-")[0]
        month = event_params["source_origin_time"].split("T")[0].split("-")[1]
        evid = event_params["source_id"].replace("hawaii", "")

        # quality check
        p_time = None
        s_time = None
        org_time = UTCDateTime(event_params["source_origin_time"])
        if not pd.isna(trace_params["trace_p_arrival_time"]):
            p_time = UTCDateTime(trace_params["trace_p_arrival_time"])
        if not pd.isna(trace_params["trace_s_arrival_time"]):
            s_time = UTCDateTime(trace_params["trace_s_arrival_time"])
        if (p_time is None) and (s_time is None):
            abnormal_traces.append(
                {**event_params, **trace_params, "remark": "No_picks"}
            )
            continue

        if (p_time is not None) and (s_time is not None):
            if p_time > s_time:
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "P>S"}
                )
                logger.warning(f"""{waveform_information} P>S""")
                continue
            elif p_time < org_time:
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "P<origin"}
                )
                logger.warning(f"""{waveform_information} P<origin""")
                continue
        elif (p_time is not None) and (s_time is None):
            if p_time < org_time:
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "P<origin"}
                )
                logger.warning(f"""{waveform_information} P<origin""")
                continue
        elif (p_time is None) and (s_time is not None):
            if s_time < org_time:
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "S<origin"}
                )
                logger.warning(f"""{waveform_information} S<origin""")
                continue

        subdir = src_dir / year / month / f"{evid}.dir"
        if not subdir.exists():
            logger.warning(f"Found no data folder for {waveform_information}")
            abnormal_traces.append(
                {**event_params, **trace_params, "remark": "No_folder"}
            )
            continue

        if subdir != last_subdir:
            last_subdir = subdir
            sts = _read_sac_files(data_dir=subdir, t_offset=36000)

        waveforms = sts.select(station=sta).copy()
        channels = [tr.stats.channel for tr in waveforms]
        if "V" in channels and "Z" in channels:
            data_with_VZ.append(
                {
                    **event_params,
                    **trace_params,
                    "remark": f"VZ",
                }
            )

    def _save_mseed_log(records, log_save_dir, fname):
        if len(records) > 0:
            df = pd.DataFrame(data=records)
            df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
            df.to_csv(log_save_dir / fname, index=False)

    _save_mseed_log(data_with_VZ, log_dir, f"VZ{process_mark}.csv")
    _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv")


def _assemble_subprocess_csvlogs(log_dir, fname, retry):
    file_name, _, file_extension = fname.rpartition(".")
    file_paths = list(log_dir.glob(f"{file_name}?*.{file_extension}"))
    if len(file_paths) > 0:
        df_chunks = [pd.read_csv(x) for x in file_paths]
        df = pd.concat(df_chunks, ignore_index=True)
        df.sort_values(by=["source_origin_time"], ignore_index=True, inplace=True)
        log_file = log_dir / fname
        if retry == False or (not log_file.exists()):
            # overwrite or create
            df.to_csv(log_file, index=False, mode="w")
        else:
            # append
            df.to_csv(log_file, index=False, header=False, mode="a")
        # delete those temperary log files genearated by subprocesses
        for tmp in file_paths:
            tmp.unlink()


def check_V_Z_parallel(
    catalog_table,
    num_processes=32,
    src_dir=None,
    dest_dir=None,
    log_level=logging.INFO,
):
    if isinstance(dest_dir, str):
        dest_dir = Path(dest_dir)
    try:
        dest_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{dest_dir} exists")
    # log_dir = self.save_dir / "sac2mseed_log"
    log_dir = dest_dir / "log"
    try:
        log_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{log_dir} exists")
    # Setting the method for multiprocessing to start new child processes
    ctx = mp.get_context("spawn")
    mp_start_method = ctx.get_start_method()
    # mp_start_method = "spawn"
    # mp.set_start_method(mp_start_method)

    print(
        f"There are {mp.cpu_count()} cpu in this machine. {num_processes} processes are used."
    )

    catalog_chunks = []
    chunksize = len(catalog_table) // num_processes
    print(f"Chunk size: {chunksize}")
    assert chunksize >= 2, (
        f"{num_processes} processes are used. Start method: {mp_start_method}. Chunk size is {chunksize}."
        f"Please try using less process"
    )
    for i in range(num_processes - 1):
        catalog_chunks.append(catalog_table.iloc[:chunksize].copy())
        catalog_table.drop(catalog_table.index[:chunksize], inplace=True)

    catalog_chunks.append(catalog_table.copy())
    catalog_table.drop(catalog_table.index[:], inplace=True)
    process_list = []
    proc_names = []
    for i in range(num_processes):
        proc_name = f"_p{i}"
        proc_names.append(proc_name)
        proc = ctx.Process(
            target=check_V_Z,
            kwargs={
                "catalog_table": catalog_chunks[i],
                "src_dir": src_dir,
                "dest_dir": dest_dir,
                "log_level": log_level,
            },
            name=proc_name,
        )
        process_list.append(proc)
    for i, proc in enumerate(process_list):
        print(f"Starting process '{proc.name}'. Chunk size: {len(catalog_chunks[i])}")
        proc.start()
    for proc in process_list:
        proc.join()
        print(f"Finished joining {proc.name}")
    ### Merge csv files generated by subprocesses
    log_files = [
        "VZ.csv",
        "abnormal_traces.csv",
    ]
    for fname in log_files:
        _assemble_subprocess_csvlogs(log_dir, fname, retry=False)


if __name__ == "__main__":
    catalog_table = pd.read_csv(
        "/mnt/DATA2/YiyuanZhong/my_data/hawaii1986to2011/sac2mseed_log/convert.csv"
    )
    check_V_Z_parallel(
        catalog_table=catalog_table,
        num_processes=32,
        src_dir="/mnt/DATA2/YiyuanZhong/LP_catalog/Hawaii/annualCUSPwaveforms",
        dest_dir="/mnt/DATA2/YiyuanZhong/my_data/hawaii1986to2011/check_VZ_channels",
    )
