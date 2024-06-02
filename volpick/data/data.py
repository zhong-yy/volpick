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
from tqdm import tqdm

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

from libcomcat.dataframes import get_phase_dataframe
from libcomcat.search import get_event_by_id
from json import JSONDecodeError

import shutil


class CatalogBase(obe.Catalog, ABC):
    """
    Base class for dataset compilation
    For parameters of obspy.core.event.Catalog, refer to
     https://docs.obspy.org/packages/autogen/obspy.core.event.Catalog.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def read_catalog(self, **kwargs):
        pass


class HinetClient2(HinetPy.Client):
    def get_event_waveform(
        self,
        starttime,
        endtime,
        region="00",
        minmagnitude=3.0,
        maxmagnitude=9.9,
        include_unknown_mag=True,
        mindepth=None,
        maxdepth=None,
        minlatitude=None,
        maxlatitude=None,
        minlongitude=None,
        maxlongitude=None,
        latitude=None,
        longitude=None,
        minradius=None,
        maxradius=None,
    ):
        # pylint: disable=too-many-arguments,too-many-locals
        starttime, endtime = to_datetime(starttime), to_datetime(endtime)

        # get event list
        events = []
        for i in range((endtime.date() - starttime.date()).days + 1):
            event_date = starttime.date() + timedelta(days=i)
            events.extend(
                self._search_event_by_day(
                    event_date.year,
                    event_date.month,
                    event_date.day,
                    region=region,
                    magmin=minmagnitude,
                    magmax=maxmagnitude,
                    include_unknown_mag=include_unknown_mag,
                )
            )

        # select events
        selected_events = []
        for event in events:
            # select events based on origin time
            if not starttime <= event.origin <= endtime:
                continue
            # select events based on magnitude
            if (event.magnitude != -99.9) and (
                not minmagnitude <= event.magnitude <= maxmagnitude
            ):
                continue
            # select events based on depth
            if mindepth is not None and event.depth < mindepth:
                continue
            if maxdepth is not None and event.depth > maxdepth:
                continue

            # select events in a box region
            if (
                minlatitude is not None
                or maxlatitude is not None
                or minlongitude is not None
                or maxlongitude is not None
            ):
                if not point_inside_box(
                    event.latitude,
                    event.longitude,
                    minlatitude=minlatitude,
                    maxlatitude=maxlatitude,
                    minlongitude=minlongitude,
                    maxlongitude=maxlongitude,
                ):
                    continue

            # select events in a circular region
            if (latitude is not None and longitude is not None) and (
                minradius is not None or maxradius is not None
            ):
                if not point_inside_circular(
                    event.latitude,
                    event.longitude,
                    latitude,
                    longitude,
                    minradius=minradius,
                    maxradius=maxradius,
                ):
                    continue
            selected_events.append(event)

        HinetPy.client.logger.info(
            "EVENT WAVEFORM DOWNLOADER| %d events to download.", len(selected_events)
        )
        dirnames = []
        for event in selected_events:
            id = self._request_event_waveform(event)
            dirname = self._download_event_waveform(id)
            if dirname is not None:
                dirnames.append(dirname)
            # HinetPy.client.logger.info("%s %s", event, dirname)
        return dirnames


class JapanDataset(CatalogBase):
    def __init__(
        self,
        save_dir=None,
        root_folder_name="Japan",
        cat_file_name="japan_catalog",
        **kwargs,
    ):
        self.root_folder_name = root_folder_name
        self.cat_file_name = cat_file_name
        self._save_dir = save_dir
        self.save_dir = self._save_dir
        self._save_quakeml_path = self.save_dir / f"{self.cat_file_name}.xml"
        self._save_csv_path = self.save_dir / f"{self.cat_file_name}.csv"
        super().__init__(**kwargs)

    @classmethod
    def download_jma_unified_catalog(
        cls,
        save_dir,
        username,
        password,
        startdate=datetime(2004, 4, 1),
        enddate=datetime(2023, 6, 30),
    ):
        client = HinetPy.Client(username, password)
        startdate_limit = enddate - timedelta(days=6)
        last_connect = time.perf_counter()
        while startdate <= startdate_limit:
            print(
                f"""{startdate.strftime("%Y%m%d")}-{(startdate + timedelta(days=6)).strftime("%Y%m%d")}"""
            )
            client.get_arrivaltime(
                startdate,
                7,
                filename=(
                    save_dir
                    + "/cat_"
                    + startdate.strftime("%Y%m%d")
                    + "_"
                    + (startdate + timedelta(days=6)).strftime("%Y%m%d")
                ),
            )
            startdate += timedelta(days=7)

            # re-connect to the data center every 10 minutes
            now = time.perf_counter()
            if now - last_connect > 10 * 60:
                client = HinetPy.Client(username, password)
                last_connect = now

    @classmethod
    def check_jma_unified_catalog(cls, catalog_dir):
        catalog_dir = Path(catalog_dir)
        catalog_files = list(catalog_dir.iterdir())
        print(f"{len(catalog_files)} files")
        flags1 = []
        for cat in catalog_files:
            with open(cat, "r") as f:
                try:
                    lines = f.readlines()
                except Exception:
                    print(f"{cat} cannot be read correctly. You may delete this file.")
                    raise Exception
            if len(lines) == 1:
                print(f"{cat} only one line")
            flags1.append(any(["<!DOCTYPE html>" in x for x in lines]))
        if any(flags1):
            ind = np.flatnonzero(flags1)
            print(f"Failed download: {catalog_files[ind]}")
            # flags2.append(any["<!DOCTYPE html>" in x for x in lines])

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if isinstance(value, str):
            value = Path(value)
        self._save_dir = value
        if self._save_dir == None:
            self._save_dir = volpick.cache_root / self.root_folder_name
            # print(f"The default saving directory is used: {self._save_dir}")
        else:
            pass
            # print(f"Set the saving directory to {self._save_dir}")

        try:
            self._save_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
            # print(f"{self._save_dir} exists")

        self._save_quakeml_path = self.save_dir / f"{self.cat_file_name}.xml"
        self._save_csv_path = self.save_dir / f"{self.cat_file_name}.csv"

    def _read_an_event(self, f, ignore_match_filtered_data=True):
        """
        Refer to "JMA: Arrival time data file format"
        https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/fmtdk_e.html
        """
        hypocentor_records = []
        current_line = f.readline()
        if not current_line:
            return None, None, None, None
        arrival_time_records = []
        comment_records = []

        while current_line:
            if current_line[0] in ["J", "U", "I"]:
                hypocentor_records.append(current_line)
            elif current_line[0] == "C":
                comment_records.append(current_line)
            elif current_line[0] == "_":
                arrival_time_records.append(current_line)
            elif current_line[0] == "W":
                if not ignore_match_filtered_data:
                    arrival_time_records.append(current_line)
            elif current_line[0] == "E":
                terminator_line = current_line
                break
            current_line = f.readline()

        return (
            hypocentor_records,
            comment_records,
            arrival_time_records,
            terminator_line,
        )

    def read(self, pathname=None, format="csv"):
        """
        Read the whole catalog. If `format` is "quakeml", the catalog is saved to
        the current object (self), and the return value is None.
        If `format` is "csv", the catalog is returned as a pandas DataFrame object.

        pathname: path to the quakeml file or csv file
        format: "quakeml" or "csv"

        return value: None (format="quakeml) or pd.DataFrame (format="csv")
        """
        if format == "quakeml":
            if pathname is None:
                pathname = self._save_quakeml_path
            cat = obe.read_events(pathname)
            self.__dict__.update(cat.copy().__dict__)
            return None
        elif format == "csv":
            if pathname is None:
                pathname = self._save_csv_path
            df = pd.read_csv(pathname)
            return df
        else:
            raise ValueError("The paramerter 'format' must be 'csv' or 'quakeml'")

    def print(self, on_screen=True, to_file=None):
        """
        Show the events and associated picks on the screen or write them into a csv file
        """
        table_items = []
        for event in self.events:
            source_id = str(event.resource_id).split("/")[-1]
            origin = event.preferred_origin()
            mag = event.preferred_magnitude()

            if origin.extra["horizontal_error"]["value"]:
                horizontal_error = float(origin.extra["horizontal_error"]["value"])
            else:
                horizontal_error = None
            if origin.extra["vertical_error"]["value"]:
                vertical_error = float(origin.extra["vertical_error"]["value"])
            else:
                vertical_error = None
            source_type = event.extra["source_type"]["value"]
            if on_screen:
                print(
                    f"""Event {source_id}: {str(origin.time):25s}| {origin.latitude:.4f} (deg), {origin.longitude:.4f} (deg), {origin.depth} (km) | Err(km): h {horizontal_error}, v {vertical_error} | Mag {mag.mag} {mag.magnitude_type} | type: {source_type}"""
                )

            station_group = defaultdict(list)
            for arrival in origin.arrivals:
                pick_obj = arrival.pick_id.get_referred_object()
                net = pick_obj.waveform_id.network_code
                sta = pick_obj.waveform_id.station_code
                loc = pick_obj.waveform_id.location_code
                cha = pick_obj.waveform_id.channel_code
                # net, sta, loc, cha = pick_obj.waveform_id.id.split(".")
                station_group[f"{sta}"].append(pick_obj)

            for sta, pick_list in station_group.items():
                net = None
                loc = "--"
                p_pick = None
                s_pick = None
                for pick0 in pick_list:
                    if pick0.phase_hint == "P":
                        p_pick = pick0.time
                    elif pick0.phase_hint == "S":
                        s_pick = pick0.time
                station_type = None
                p_flag = None
                s_flag = None
                if len(pick_list) > 0:
                    station_type = pick_list[0].extra["station_type"]["value"]
                    p_flag = pick_list[0].extra["trace_p_flag"]["value"]
                    s_flag = pick_list[0].extra["trace_s_flag"]["value"]
                disp_msg = f"""{sta:5s} |Seismometer type: {station_type}| P: {p_pick}| S: {s_pick}"""

                if to_file:
                    table_items.append(
                        {
                            "source_id": source_id,
                            "source_origin_time": str(origin.time),
                            "source_latitude_deg": origin.latitude,
                            "source_longitude_deg": origin.longitude,
                            "source_depth_km": origin.depth,
                            "source_magnitude": mag.mag,
                            "source_magnitude_type": mag.magnitude_type,
                            "source_type": source_type,
                            "station_network_code": net,
                            "station_code": sta,
                            "station_location_code": loc,
                            "trace_channel": cha,
                            "station_type": station_type,
                            "trace_p_arrival_time": p_pick,
                            "trace_s_arrival_time": s_pick,
                            "trace_p_flag": p_flag,
                            "trace_s_flag": s_flag,
                        }
                    )
                if on_screen:
                    print(disp_msg)
        if to_file:
            table = pd.DataFrame(table_items)
            table.to_csv(to_file, index=False)

    def read_catalog_multiple_files(
        self, catalog_dir, id_prefix="", num_processes=24, delete_temp_files=True
    ):
        # Setting the method for multiprocessing to start new child processes
        ctx = mp.get_context("spawn")
        mp_start_method = ctx.get_start_method()
        # mp.set_start_method(mp_start_method)

        catalog_dir = Path(catalog_dir)
        catalog_files = list(catalog_dir.iterdir())
        catalog_files.sort()
        print(
            f"There are {mp.cpu_count()} cpu in this machine. {num_processes} processes are used. mp start_method={mp_start_method}"
        )
        cat_chunks = []
        chunksize = len(catalog_files) // num_processes
        for i in range(num_processes - 1):
            cat_chunks.append(catalog_files[i * chunksize : (i + 1) * chunksize].copy())
        cat_chunks.append(catalog_files[(num_processes - 1) * chunksize :])

        process_list = []
        proc_names = []
        for i in range(num_processes):
            proc_name = f"_p{i}"
            proc_names.append(proc_name)
            proc = ctx.Process(
                target=self._read_catalog_worker,
                kwargs={
                    "save_dir": self.save_dir,
                    "id_prefix": id_prefix,
                    "station_archive_file_list": cat_chunks[i],
                },
                name=proc_name,
            )
            process_list.append(proc)
        for i, proc in enumerate(process_list):
            print(f"Starting process '{proc.name}'. Chunk size: {len(cat_chunks[i])}")
            proc.start()
        for proc in process_list:
            proc.join()
            print(f"Finished joining {proc.name}")

        def _merge_temp_files(pattern_str, save_path):
            chunk_csv_files = list(self.save_dir.glob(pattern_str))
            chunk_csv_files.sort()
            if len(chunk_csv_files) > 0:
                df_chunks = [pd.read_csv(x) for x in chunk_csv_files]
                df = pd.concat(df_chunks, ignore_index=True)
                df.to_csv(save_path, index=False, mode="w")
                # delete those temperary log files genearated by subprocesses
                if delete_temp_files:
                    for tmp in chunk_csv_files:
                        tmp.unlink()

        print("Merging files. It will take a few minutes ...")
        _merge_temp_files(
            pattern_str="_*_skipped_events.csv",
            save_path=self.save_dir / "skipped_events.csv",
        )
        _merge_temp_files(
            pattern_str="_*_skipped_stations.csv",
            save_path=self.save_dir / "skipped_stations.csv",
        )

        _merge_temp_files(
            pattern_str="_*_arrival_times.csv",
            save_path=self._save_csv_path,
        )

    def _read_catalog_worker(self, save_dir, id_prefix, station_archive_file_list):
        for station_archive_file in station_archive_file_list:
            japan_cat = JapanDataset(save_dir=save_dir)
            japan_cat._save_csv_path = (
                self.save_dir / f"_{station_archive_file.name}_arrival_times.csv"
            )
            skipped_events_log = (
                self.save_dir / f"_{station_archive_file.name}_skipped_events.csv"
            )
            skipped_stations_log = (
                self.save_dir / f"_{station_archive_file.name}_skipped_stations.csv"
            )
            # print(japan_cat._save_csv_path)
            japan_cat.read_catalog(
                station_archive_file=station_archive_file,
                id_prefix=id_prefix,
                save_csv=True,
                save_quakeml=False,
                on_screen=False,
                skipped_events_log=skipped_events_log,
                skipped_stations_log=skipped_stations_log,
            )

    def read_catalog(
        self,
        station_archive_file,
        n_events=None,
        id_prefix="",
        min_date=UTCDateTime("1900-01-01T00:00:00.0"),
        max_date=UTCDateTime("2999-01-01T00:00:00.0"),
        save_quakeml=False,
        save_csv=False,
        on_screen=True,
        skipped_events_log=None,
        skipped_stations_log=None,
    ):
        skipped_events = []
        skipped_stations = []
        if n_events is None:
            n_events = np.inf
        with open(station_archive_file, "r") as f:
            i = 0
            while i < n_events:
                hypocentor_records, _, station_lines, _ = self._read_an_event(f)
                if hypocentor_records is None:  # if it is empty
                    break
                summary_line = hypocentor_records[0]
                event_id = (
                    id_prefix
                    + summary_line[0]
                    + summary_line[3:17].strip()
                    + summary_line[21:28].replace(" ", "")
                    + summary_line[32:40].replace(" ", "")
                )
                o_year = summary_line[1:5]
                o_month = summary_line[5:7]
                o_date = summary_line[7:9]
                o_hour = summary_line[9:11]
                o_min = summary_line[11:13]
                o_sec = summary_line[13:15] + "." + summary_line[15:17]
                origin_time_str = (
                    o_year
                    + "-"
                    + o_month
                    + "-"
                    + o_date
                    + "T"
                    + o_hour
                    + ":"
                    + o_min
                    + ":"
                    + o_sec
                )
                origin_time = UTCDateTime(origin_time_str)

                if origin_time.timestamp < min_date.timestamp:
                    continue
                if origin_time.timestamp > max_date.timestamp:
                    break

                if (
                    len(summary_line[21:28].strip()) == 0
                    or len(summary_line[32:40].strip()) == 0
                ):
                    skipped_events.append(
                        {
                            "Hypocenter_record": summary_line.replace("\n", ""),
                            "remark": "Empty location",
                        }
                    )
                    continue

                try:
                    hy_lat = (
                        float(summary_line[21:24].strip())
                        + float(summary_line[24:28].strip()) / 100.0 / 60.0
                    )
                    hy_lon = (
                        float(summary_line[32:36].strip())
                        + float(summary_line[36:40].strip()) / 100.0 / 60.0
                    )
                    hy_dep_str = summary_line[44:49]
                    if hy_dep_str[3:5] == "  ":
                        hy_dep_str = hy_dep_str[:3]
                        # hy_dep_str = hy_dep_str.replace(" ", "0")
                        hy_dep = float(hy_dep_str)
                    else:
                        # hy_dep_str = hy_dep_str.replace(" ", "0")
                        hy_dep = float(hy_dep_str) / 100.0
                except Exception:
                    skipped_events.append(
                        {
                            "Hypocenter_record": summary_line.replace("\n", ""),
                            "remark": "Uncertain lat/lon/dep format",
                        }
                    )
                    continue

                org = obe.Origin()
                org.time = origin_time
                org.latitude = hy_lat  # unit: deg
                org.longitude = hy_lon  # unit: deg
                org.depth = hy_dep  # unit: km

                horizontal_error = None
                vertical_error = None
                org_extra = {
                    "horizontal_error": {
                        "value": horizontal_error,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    },
                    "vertical_error": {
                        "value": vertical_error,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    },
                }
                org.extra = org_extra

                # magnitude
                ev_mag = None
                mag_type = None
                if summary_line[52:54].strip():
                    ev_mag_str = (
                        summary_line[52:54].replace("A", "-1").replace("B", "-2")
                    )
                    ev_mag = float(ev_mag_str) / 10
                if summary_line[54].strip():
                    mag_type = summary_line[54].strip()
                mag = obe.Magnitude(mag=ev_mag, magnitude_type=mag_type)
                mag.origin_id = org.resource_id

                event = obe.Event(resource_id=obe.ResourceIdentifier(id=event_id))
                event.origins.append(org)
                event.preferred_origin_id = event.origins[-1].resource_id
                event.magnitudes.append(mag)
                event.preferred_magnitude_id = event.magnitudes[-1].resource_id
                ev_type_dict = {
                    "1": "natural",
                    "2": "insufficient",
                    "3": "artificial",
                    "4": "eruption",
                    "5": "lp",
                }
                if summary_line[60].strip() in ev_type_dict:
                    event_type = ev_type_dict[summary_line[60].strip()]
                else:
                    event_type = "unknown"
                    skipped_events.append(
                        {
                            "Hypocenter_record": summary_line.replace("\n", ""),
                            "remark": "unkown type of event",
                        }
                    )
                    continue

                event_extra = {
                    "source_type": {
                        "value": event_type,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    }
                }

                def get_phase_label(phase_name):
                    # if "P" in phase_name:
                    if phase_name.strip() in ["IP", "EP", "P"]:
                        return "P"
                    elif phase_name.strip() in ["IS", "ES", "S"]:
                        return "S"
                    else:
                        return None

                event.extra = event_extra
                for station_line in station_lines:
                    sta = station_line[1:7].strip()
                    net = None
                    cha = "--"
                    loc = "--"

                    arr_date = station_line[13:15].strip()
                    if arr_date[0] == " ":
                        arr_date = "0" + arr_date[1]

                    arr_year = station_line[87:89]
                    if arr_year[0] == " ":
                        arr_year = "0" + arr_year[1]
                    arr_year = summary_line[1:3] + arr_year

                    arr_mon = station_line[89:91].strip()
                    if arr_mon[0] == " ":
                        arr_mon = "0" + arr_mon[1]

                    phase_name1 = station_line[15:19].strip()
                    phase_label1 = None
                    phase_name2 = station_line[27:31].strip()
                    phase_label2 = None
                    ps_picks = {"P": None, "S": None}
                    seismometer_type = None
                    if station_line[12].strip():
                        seismometer_type = station_line[12]
                    p_flag = station_line[91]
                    s_flag = station_line[92]
                    pick_extra = {
                        "station_type": {
                            "value": seismometer_type,
                            "namespace": "http://some-page.de/xmlns/1.0",
                        },
                        "trace_p_flag": {
                            "value": p_flag,
                            "namespace": "http://some-page.de/xmlns/1.0",
                        },
                        "trace_s_flag": {
                            "value": s_flag,
                            "namespace": "http://some-page.de/xmlns/1.0",
                        },
                    }
                    if phase_name1 and summary_line[19:27].strip():
                        phase_label1 = get_phase_label(phase_name1)
                        if phase_label1 is None:
                            skipped_stations.append(
                                {
                                    "Hypocenter_record": summary_line.replace("\n", ""),
                                    "arrival_time_record": station_line.replace(
                                        "\n", ""
                                    ),
                                    "remark": f"Unknown phase name: {phase_name1.strip()}",
                                }
                            )
                            continue
                        arr_hr = station_line[19:21].strip()
                        arr_min = station_line[21:23].strip()
                        arr_sec = station_line[23:25] + "." + station_line[25:27]
                        try:
                            arr1 = UTCDateTime(
                                arr_year
                                + "-"
                                + arr_mon
                                + "-"
                                + arr_date
                                + "T"
                                + arr_hr
                                + ":"
                                + arr_min
                                + ":00.0"
                            ) + float(arr_sec)
                        except Exception:
                            skipped_stations.append(
                                {
                                    "Hypocenter_record": summary_line.replace("\n", ""),
                                    "arrival_time_record": station_line.replace(
                                        "\n", ""
                                    ),
                                    "remark": f"Unknown format",
                                }
                            )
                            continue
                        ps_picks[phase_label1] = arr1

                        if phase_name2 and summary_line[31:37]:
                            phase_label2 = get_phase_label(phase_name2)
                            if phase_label2 is None:  # if label is not P or S, e.g. M
                                skipped_stations.append(
                                    {
                                        "Hypocenter_record": summary_line.replace(
                                            "\n", ""
                                        ),
                                        "arrival_time_record": station_line.replace(
                                            "\n", ""
                                        ),
                                        "remark": f"Unknown phase name: {phase_name2.strip()}",
                                    }
                                )
                                continue

                            if phase_label2 != phase_label1:
                                arr_min2 = station_line[31:33].strip()
                                arr_sec2 = (
                                    station_line[33:35] + "." + station_line[35:37]
                                )
                                try:
                                    arr2 = UTCDateTime(
                                        arr_year
                                        + "-"
                                        + arr_mon
                                        + "-"
                                        + arr_date
                                        + "T"
                                        + arr_hr
                                        + ":"
                                        + arr_min2
                                        + ":00.0"
                                    ) + float(arr_sec2)
                                except Exception:
                                    skipped_stations.append(
                                        {
                                            "Hypocenter_record": summary_line.replace(
                                                "\n", ""
                                            ),
                                            "arrival_time_record": station_line.replace(
                                                "\n", ""
                                            ),
                                            "remark": f"Unknown format",
                                        }
                                    )
                                    continue
                                ps_picks[phase_label2] = arr2
                    else:
                        skipped_stations.append(
                            {
                                "Hypocenter_record": summary_line.replace("\n", ""),
                                "arrival_time_record": station_line.replace("\n", ""),
                                "remark": "Empty phase",
                            }
                        )
                        continue

                    p_pick = ps_picks["P"]
                    s_pick = ps_picks["S"]
                    if p_pick:
                        p_pick_obj = obe.Pick(
                            waveform_id=obe.WaveformStreamID(
                                network_code=net,
                                station_code=sta,
                                location_code=loc,
                                channel_code=cha,
                            ),
                            time=p_pick,
                            phase_hint="P",
                        )
                        p_pick_obj.extra = pick_extra
                        p_arrival_obj = obe.Arrival(pick_id=p_pick_obj.resource_id)
                        event.picks.append(p_pick_obj)
                        event.preferred_origin().arrivals.append(p_arrival_obj)
                    if s_pick:
                        s_pick_obj = obe.Pick(
                            waveform_id=obe.WaveformStreamID(
                                network_code=net,
                                station_code=sta,
                                location_code=loc,
                                channel_code=cha,
                            ),
                            time=s_pick,
                            phase_hint="S",
                        )
                        s_pick_obj.extra = pick_extra
                        s_arrival_obj = obe.Arrival(pick_id=s_pick_obj.resource_id)
                        event.picks.append(s_pick_obj)
                        event.preferred_origin().arrivals.append(s_arrival_obj)

                if len(event.picks) > 0:
                    self.events.append(event)
                    i = i + 1
        if skipped_events_log:
            if len(skipped_events) > 0:
                df = pd.DataFrame(data=skipped_events)
                df.to_csv(skipped_events_log, index=False)
        if skipped_stations_log:
            if len(skipped_stations) > 0:
                df = pd.DataFrame(data=skipped_stations)
                df.to_csv(skipped_stations_log, index=False)

        if save_quakeml:
            print("Writing to quakeml format ...")
            self.write(self._save_quakeml_path, format="QUAKEML")
        if save_csv is not None:
            print(f"Writing to csv {self._save_csv_path} ...")
            self.print(on_screen=on_screen, to_file=self._save_csv_path)
        elif on_screen:
            self.print(on_screen=on_screen, to_file=False)

    def _read_sac_files(self, data_dir):
        sac_files = list(data_dir.glob("*.SAC"))
        sts = Stream()
        for sac in sac_files:
            st = read(sac)
            sts += st
        return sts

    def _assemble_subprocess_csvlogs(self, log_dir, fname, append_to_file):
        file_name, _, file_extension = fname.rpartition(".")
        file_paths = list(log_dir.glob(f"{file_name}?*.{file_extension}"))
        if len(file_paths) > 0:
            df_chunks = [pd.read_csv(x) for x in file_paths]
            df = pd.concat(df_chunks, ignore_index=True)
            df.sort_values(by=["source_origin_time"], ignore_index=True, inplace=True)
            log_file = log_dir / fname
            if append_to_file == False or (not log_file.exists()):
                # overwrite or create
                df.to_csv(log_file, index=False, mode="w")
            else:
                # append
                df.to_csv(log_file, index=False, header=False, mode="a")
            # delete those temperary log files genearated by subprocesses
            for tmp in file_paths:
                tmp.unlink()

    def download_data(
        self,
        catalog_table,
        username,
        password,
        sampling_rate=None,
        num_processes=3,
        download_dir=None,
        win_len_lim=5,
        log_level=logging.INFO,
        delete_temp_files=False,
        retry=False,
    ):
        print(
            "Warning: parallelly downloading Japanese seismic data is prone to network error and misssing data. The errors are difficult to track. Please try _download(...) function first"
        )
        if download_dir is None:
            download_dir = self.save_dir / "mseed"
        if isinstance(download_dir, str):
            download_dir = Path(download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{download_dir} exists")

        log_dir = self.save_dir / "mseed_log"
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{log_dir} exists")

        # Setting the method for multiprocessing to start new child processes
        ctx = mp.get_context("spawn")
        mp_start_method = ctx.get_start_method()
        # mp.set_start_method(mp_start_method)

        print(
            f"There are {mp.cpu_count()} cpu in this machine. {num_processes} processes are used."
        )
        event_ids = np.unique(catalog_table["source_id"])
        catalog_chunks = []
        chunksize = len(event_ids) // num_processes
        print(f"~ {chunksize} events in a chunk")
        assert chunksize >= 2, (
            f"{num_processes} processes are used. Start method: {mp_start_method}. Chunk size is {chunksize}."
            f"Please try using less process"
        )
        for i in range(num_processes):
            if i == num_processes - 1:
                catalog_chunks.append(
                    catalog_table[
                        catalog_table["source_id"].isin(event_ids[i * chunksize :])
                    ].copy()
                )
            else:
                catalog_chunks.append(
                    catalog_table[
                        catalog_table["source_id"].isin(
                            event_ids[i * chunksize : (i + 1) * chunksize]
                        )
                    ].copy()
                )
        # for i in range(num_processes - 1):
        #     catalog_chunks.append(catalog_table.iloc[:chunksize].copy())
        #     catalog_table.drop(catalog_table.index[:chunksize], inplace=True)
        # catalog_chunks.append(catalog_table.copy())
        catalog_table.drop(catalog_table.index[:], inplace=True)

        process_list = []
        proc_names = []
        for i in range(num_processes):
            proc_name = f"_p{i}"
            proc_names.append(proc_name)
            proc = ctx.Process(
                target=self._download,
                kwargs={
                    "catalog_table": catalog_chunks[i],
                    "username": username,
                    "password": password,
                    "sampling_rate": sampling_rate,
                    "download_dir": download_dir,
                    "win_len_lim": win_len_lim,
                    "log_level": log_level,
                    "delete_temp_files": delete_temp_files,
                },
                name=proc_name,
            )
            process_list.append(proc)
        for i, proc in enumerate(process_list):
            print(
                f"Starting process '{proc.name}'. Chunk size: {len(catalog_chunks[i])}"
            )
            proc.start()
        for proc in process_list:
            proc.join()
            print(f"Finished joining {proc.name}")

        ### Merge csv files generated by subprocesses
        # log_files = [
        #     "downloads.csv",
        #     "failed_downloads.csv",
        #     "abnormal_traces.csv",
        # ]
        # for fname in log_files:
        #     self._assemble_subprocess_csvlogs(log_dir, fname, retry)

        self._assemble_subprocess_csvlogs(log_dir, "downloads.csv", retry)
        self._assemble_subprocess_csvlogs(log_dir, "abnormal_traces.csv", retry)

        # failed_downloads.csv is always overwritten, no mattter what "retry" is
        self._assemble_subprocess_csvlogs(
            log_dir, "failed_downloads.csv", append_to_file=False
        )
        self._assemble_subprocess_csvlogs(
            log_dir, "multiple_dirs.csv", append_to_file=False
        )

    def _download(
        self,
        catalog_table,
        username,
        password,
        sampling_rate=None,
        download_dir=None,
        win_len_lim=5,
        log_level=logging.INFO,
        delete_temp_files=False,
    ):
        def _save_mseed_log(records, log_save_dir, fname):
            if len(records) > 0:
                df = pd.DataFrame(data=records)
                df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
                df.to_csv(log_save_dir / fname, index=False)

        # If the process is the main process (sequential downloading),
        # first check and create download_dir and log_dir
        if mp.parent_process() is None:  # check whether it is the main process
            if download_dir is None:
                download_dir = self.save_dir / "mseed"
            if isinstance(download_dir, str):
                download_dir = Path(download_dir)
            try:
                download_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"{download_dir} exists")
        # log_dir = self.save_dir / "mseed_log"
        log_dir = download_dir.parent / f"{download_dir.name}_log"
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
        logger = volpick.logger.getChild("download" + process_mark)
        logger.setLevel(log_level)

        successful_downloads = []
        multiple_downloaded_dirs = []
        abnormal_traces = []
        failed_downloads = []
        event_ids = np.unique(catalog_table["source_id"])

        def connect_client():
            while True:
                try:
                    client = HinetClient2(user=username, password=password, retries=5)
                    logger.info("Connected successfully")
                except Exception as e:
                    logger.error(
                        f"{type(e).__name__}. Waiting for 3s before next try ..."
                    )
                    time.sleep(3)
                    continue
                else:
                    break
            last_connect = time.perf_counter()
            return client, last_connect

        client, last_connect = connect_client()

        for idx, ev_id in enumerate(event_ids):
            logger.info(f"""Processing the {idx}th event in {len(event_ids)} events""")
            event_waveforms = catalog_table[catalog_table["source_id"] == ev_id]
            event_params = {
                "source_id": event_waveforms.iloc[0]["source_id"],
                "source_origin_time": event_waveforms.iloc[0]["source_origin_time"],
                "source_latitude_deg": event_waveforms.iloc[0]["source_latitude_deg"],
                "source_longitude_deg": event_waveforms.iloc[0]["source_longitude_deg"],
                "source_depth_km": event_waveforms.iloc[0]["source_depth_km"],
                "source_magnitude": event_waveforms.iloc[0]["source_magnitude"],
                "source_magnitude_type": event_waveforms.iloc[0][
                    "source_magnitude_type"
                ],
                "source_type": event_waveforms.iloc[0]["source_type"],
            }

            org_time = UTCDateTime(event_params["source_origin_time"])
            dirnames = []
            # re-connect to the data center every 5 minutes
            now = time.perf_counter()
            if now - last_connect > 5 * 60:
                client, last_connect = connect_client()
            for _ in range(5):
                while True:
                    try:
                        dirnames = client.get_event_waveform(
                            datetime.fromisoformat((org_time - 0.5).isoformat()),
                            datetime.fromisoformat((org_time + 0.5).isoformat()),
                            minmagnitude=event_params["source_magnitude"] - 0.1,
                            maxmagnitude=event_params["source_magnitude"] + 0.1,
                            mindepth=event_params["source_depth_km"] - 2,
                            maxdepth=event_params["source_depth_km"] + 2,
                            minlatitude=event_params["source_latitude_deg"] - 0.02,
                            maxlatitude=event_params["source_latitude_deg"] + 0.02,
                            minlongitude=event_params["source_longitude_deg"] - 0.02,
                            maxlongitude=event_params["source_longitude_deg"] + 0.02,
                        )
                    except Exception as e:
                        logger.error(f"{type(e).__name__}| Retry")
                        client, last_connect = connect_client()
                    else:
                        break
                if len(dirnames) == 0:
                    client, last_connect = connect_client()
                else:
                    # assert len(dirnames) == 1
                    break
            if len(dirnames) > 1:
                multiple_downloaded_dirs.append(
                    {
                        **event_params,
                        "len(dirnames)": len(dirnames),
                    }
                )
                logger.warning(
                    f"Multiple events are seleted when searching for"
                    f' event {event_params["source_id"]} | {row.station_code}. Ignore this event.'
                )
                continue
            sts = Stream()
            if len(dirnames) > 0:
                data = f"{dirnames[0]}/{dirnames[0]}.evt"
                ctable = f"{dirnames[0]}/{dirnames[0]}.ch"
                try:
                    HinetPy.win32.extract_sac(
                        data,
                        ctable,
                        filter_by_name=list(event_waveforms["station_code"]),
                        outdir=dirnames[0],
                    )
                    sts = self._read_sac_files(Path(dirnames[0]))
                except Exception as e:
                    logger.error(f"{type(e).__name__} while extracting sac files")
            for row in event_waveforms.itertuples(name="trace"):
                trace_params = {
                    "station_network_code": row.station_network_code,
                    "station_code": row.station_code,
                    "station_location_code": row.station_location_code,
                    "trace_channel": row.trace_channel,
                    "trace_p_arrival_time": row.trace_p_arrival_time,
                    "trace_s_arrival_time": row.trace_s_arrival_time,
                }
                if len(sts) == 0:
                    logger.warning(
                        f"No waveforms have been downloaded "
                        f' for event {event_params["source_id"]} | {row.station_code}'
                    )
                    failed_downloads.append(
                        {**event_params, **trace_params, "remark": "No_waveforms"}
                    )
                    continue
                waveform_information = (
                    f"""{event_params["source_id"]}: {event_params["source_origin_time"]} | """
                    f"{trace_params['station_code']} |"
                )
                sta = row.station_code
                p_time = None
                s_time = None
                if not pd.isna(trace_params["trace_p_arrival_time"]):
                    p_time = UTCDateTime(trace_params["trace_p_arrival_time"])
                if not pd.isna(trace_params["trace_s_arrival_time"]):
                    s_time = UTCDateTime(trace_params["trace_s_arrival_time"])
                # quality check for phases
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

                waveforms = sts.select(station=sta).copy()
                if len(waveforms) == 0:
                    logger.warning(
                        f"Found no waveforms for {waveform_information}"
                        f' in event {event_params["source_id"]}'
                    )
                    abnormal_traces.append(
                        {**event_params, **trace_params, "remark": "No_waveforms"}
                    )
                    continue

                if np.all(
                    [
                        x.stats.channel in ["NA", "EA", "UA", "NB", "EB", "UB"]
                        for x in waveforms
                    ]
                ):
                    waveforms = waveforms.select(channel="[ENU]A").copy()
                    for tr in waveforms:
                        tr.stats.channel = tr.stats.channel.replace("EA", "E")
                        tr.stats.channel = tr.stats.channel.replace("NA", "N")
                        tr.stats.channel = tr.stats.channel.replace("UA", "U")

                for tr in waveforms:
                    if "UD" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("UD", "Z")
                    if "EW" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("EW", "E")
                    if "NS" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("NS", "N")
                    if "X" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("X", "E")
                    if "Y" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("Y", "N")
                    if "U" in tr.stats.channel:
                        tr.stats.channel = tr.stats.channel.replace("U", "Z")

                if np.any(["V" == x.stats.channel[-1:] for x in waveforms]) and np.all(
                    ["Z" != x.stats.channel[-1:] for x in waveforms]
                ):
                    for tr in waveforms:
                        if tr.stats.channel[-1] == "V":
                            tr.stats.channel = tr.stats.channel[:-1] + "Z"

                waveforms = waveforms.select(channel="[ENZ]").copy()

                if len(waveforms) == 0:
                    logger.warning(
                        f"Found no waveforms for {waveform_information}"
                        f' in event {event_params["source_id"]}'
                    )
                    abnormal_traces.append(
                        {**event_params, **trace_params, "remark": "No_waveforms"}
                    )
                    continue
                sta_lat = waveforms[0].stats.sac.get("stla", np.nan)
                sta_lon = waveforms[0].stats.sac.get("stlo", np.nan)
                sta_elev = waveforms[0].stats.sac.get("stel", np.nan)
                if not np.isnan(sta_lat * sta_lon):
                    dist, azimuth, back_azimuth = gps2dist_azimuth(
                        event_params["source_latitude_deg"],
                        event_params["source_longitude_deg"],
                        sta_lat,
                        sta_lon,
                    )
                else:
                    dist = np.nan
                    azimuth = np.nan
                    back_azimuth = np.nan
                trace_params.update(
                    {
                        "station_latitude_deg": sta_lat,
                        "station_longitude_deg": sta_lon,
                        "station_elevation_m": sta_elev,
                        "station_epicentral_distance_m": dist,
                        "path_azimuth_deg": azimuth,
                        "path_back_azimuth_deg": back_azimuth,
                    }
                )

                # length limit
                min_starttime = min(trace.stats.starttime for trace in waveforms)
                max_endtime = max(trace.stats.endtime for trace in waveforms)
                if max_endtime - min_starttime < win_len_lim:
                    logger.warning(
                        f"Window length is smaller than {win_len_lim} for {waveform_information}"
                        f' in event {event_params["source_id"]}'
                    )
                    abnormal_traces.append(
                        {
                            **event_params,
                            **trace_params,
                            "remark": f"length<{win_len_lim}s",
                        }
                    )
                    continue

                # If any of the available manual picks is located in a gap, skip it
                arrivals = [xx for xx in [p_time, s_time] if xx is not None]
                check_phases_out_of_traces = [[] for _ in range(len(arrivals))]
                for component in "ZNE":
                    c_stream = waveforms.select(channel=f"*{component}")
                    for arr_i, pha_arr_t in enumerate(arrivals):
                        for trace in c_stream:
                            check_phases_out_of_traces[arr_i].append(
                                pha_arr_t < trace.stats.starttime
                                or pha_arr_t > trace.stats.endtime
                            )
                check_phases_out_of_traces = [
                    np.all(x) for x in check_phases_out_of_traces
                ]
                if np.any(check_phases_out_of_traces):
                    logger.warning(
                        f"P and S are not within the traces for {waveform_information}"
                        f' in event {event_params["source_id"]}'
                    )
                    abnormal_traces.append(
                        {**event_params, **trace_params, "remark": "phases_in_gap"}
                    )
                    continue
                if sampling_rate is None:
                    sampling_rate = waveforms[0].stats.sampling_rate
                if any(
                    trace.stats.sampling_rate != sampling_rate for trace in waveforms
                ):
                    logger.warning(
                        f"{waveform_information} Resampling traces to common sampling rate {sampling_rate}."
                    )
                    waveforms.resample(sampling_rate)
                trace_params["trace_name"] = (
                    f"{event_params['source_id']}_{trace_params['station_code']}.mseed"
                )
                successful_downloads.append(
                    {
                        **event_params,
                        **trace_params,
                        "trace_sampling_rate_hz": sampling_rate,
                    }
                )
                waveforms.write(
                    download_dir / f"{trace_params['trace_name']}",
                    format="MSEED",
                )
                logger.info(f"""{waveform_information} successfully downloaded""")
            if delete_temp_files:
                if dirnames > 0:
                    shutil.rmtree(dirnames[0])
            if (idx + 1) % 2000 == 0:
                _save_mseed_log(
                    successful_downloads, log_dir, f"downloads{process_mark}.csv"
                )
                _save_mseed_log(
                    failed_downloads, log_dir, f"failed_downloads{process_mark}.csv"
                )
                _save_mseed_log(
                    abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv"
                )
                _save_mseed_log(
                    multiple_downloaded_dirs,
                    log_dir,
                    f"multiple_dirs{process_mark}.csv",
                )

        _save_mseed_log(successful_downloads, log_dir, f"downloads{process_mark}.csv")
        _save_mseed_log(
            failed_downloads, log_dir, f"failed_downloads{process_mark}.csv"
        )
        _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv")
        _save_mseed_log(
            multiple_downloaded_dirs,
            log_dir,
            f"multiple_dirs{process_mark}.csv",
        )


class JapanNoiseData:
    def __init__(self, save_dir=None, root_folder_name="Noise"):
        self.root_folder_name = root_folder_name
        self._save_dir = save_dir
        self.save_dir = self._save_dir

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if isinstance(value, str):
            value = Path(value)
        self._save_dir = value
        if self._save_dir == None:
            self._save_dir = volpick.cache_root / self.root_folder_name
        else:
            pass

        try:
            self._save_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

    def create_noise_table(
        self,
        base_catalog,
        time_difference_limit=2400,
        output_filename="japan_noise_reference_time.csv",
    ):
        events = base_catalog.drop_duplicates(
            subset="source_id", keep="first", ignore_index=True
        ).copy()

        events["source_origin_time"] = events["source_origin_time"].apply(
            lambda x: UTCDateTime(x) if pd.notna(x) else np.nan
        )
        events.sort_values(by=["source_origin_time"], inplace=True)
        events["event_end"] = events["source_origin_time"] + 600

        forward_event_time_diff = (
            events["source_origin_time"].values[1:] - events["event_end"].values[:-1]
        ).astype(float)

        forward_event_time_diff = np.append(forward_event_time_diff, np.nan)
        events["forward_event_time_difference"] = forward_event_time_diff
        events["next_event_origin_time"] = np.append(
            events["source_origin_time"][1:].values, np.nan
        )

        events = events.sort_values(
            by=["forward_event_time_difference"], ascending=False, inplace=False
        )
        events = events[
            pd.notna(events["forward_event_time_difference"])
            & (events["forward_event_time_difference"] > time_difference_limit)
        ]
        events.to_csv(self.save_dir / output_filename, index=False)
        print(f"Output: {self.save_dir/output_filename}")

    def download(
        self,
        catalog_table,
        stations,
        username,
        password,
        sampling_rate=None,
        download_dir=None,
        time_window=2,  # unit: minute
        win_len_lim=30,
        log_level=logging.INFO,
        tmp_dir="tmp_jp_noise",
        delete_temp_files=False,
    ):
        def _read_sac_files(data_dir):
            sac_files = list(data_dir.glob("*.SAC"))
            sts = Stream()
            for sac in sac_files:
                st = read(sac)
                sts += st
            return sts

        def _save_mseed_log(records, log_save_dir, fname):
            if len(records) > 0:
                df = pd.DataFrame(data=records)
                df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
                df.to_csv(log_save_dir / fname, index=False)

        if download_dir is None:
            download_dir = self.save_dir / "mseed"
        if isinstance(download_dir, str):
            download_dir = Path(download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{download_dir} exists")
        log_dir = download_dir.parent / f"{download_dir.name}_log"
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{log_dir} exists")
        # Initialize a logger
        logger = volpick.logger.getChild("download")
        logger.setLevel(log_level)

        successful_downloads = []
        abnormal_traces = []
        failed_downloads = []

        def connect_client():
            while True:
                try:
                    client = HinetPy.Client(user=username, password=password, retries=5)
                    logger.info("Connected successfully")
                except Exception as e:
                    logger.error(
                        f"{type(e).__name__}. Waiting for 3s before next try ..."
                    )
                    time.sleep(3)
                    continue
                else:
                    break
            last_connect = time.perf_counter()
            return client, last_connect

        client, last_connect = connect_client()
        for row in catalog_table.itertuples(name="event"):
            event_end = UTCDateTime(row.event_end)
            next_event_start = UTCDateTime(row.next_event_origin_time)
            # t_starts = [
            #     event_end + 0.5 * (next_event_start - event_end) - 150,
            #     event_end + 0.5 * (next_event_start - event_end) + 30,
            # ]
            t_starts = [
                event_end + 0.5 * (next_event_start - event_end) - 420,
                event_end + 0.5 * (next_event_start - event_end) + 300,
            ]
            event_params = {
                "source_id": None,
                "source_origin_time": None,
                "source_latitude_deg": None,
                "source_longitude_deg": None,
                "source_depth_km": None,
                "source_magnitude": None,
                "source_magnitude_type": None,
                "source_type": "noise",
            }

            for t_start in t_starts:
                # re-connect to the data center every 5 minutes
                now = time.perf_counter()
                if now - last_connect > 5 * 60:
                    client, last_connect = connect_client()
                data = None
                ctable = None
                for _ in range(2):
                    try:
                        data, ctable = client.get_continuous_waveform(
                            "0101",
                            starttime=datetime.fromisoformat(t_start.isoformat()),
                            span=time_window,
                            outdir=tmp_dir,
                        )
                    except Exception as e:
                        logger.error(f"{type(e).__name__}| Retry")
                        client, last_connect = connect_client()
                    else:
                        break

                sts = Stream()
                if data is None:
                    logger.warning(
                        f"No waveforms have been downloaded "
                        f" from {t_start} to {t_start+120}"
                    )
                    failed_downloads.append(
                        {**event_params, "starttime": t_start, "remark": "No_waveforms"}
                    )
                    continue
                else:
                    try:
                        HinetPy.win32.extract_sac(
                            data,
                            ctable,
                            filter_by_name=stations,
                            outdir=data.split(".")[0],
                        )
                        sts = _read_sac_files(Path(data.split(".")[0]))
                    except Exception as e:
                        logger.error(f"{type(e).__name__} while extracting sac files")
                        logger.warning(
                            f"Failed to extract waveforms "
                            f" from {t_start} to {t_start+120}"
                        )
                        failed_downloads.append(
                            {
                                **event_params,
                                "starttime": t_start,
                                "remark": "Failed_extraction",
                            }
                        )
                        continue
                for sta in stations:
                    trace_params = {
                        "station_network_code": None,
                        "station_code": sta,
                        "station_location_code": "--",
                        "trace_channel": "--",
                        "trace_p_arrival_time": None,
                        "trace_s_arrival_time": None,
                        "trace_p_max_weight": None,
                        "trace_s_max_weight": None,
                        "trace_p_first_motion": None,
                    }
                    waveforms = sts.select(station=sta).copy()
                    if len(waveforms) == 0:
                        logger.warning(
                            f"Found no waveforms for station {sta} from {t_start} to {t_start+120}"
                        )
                        abnormal_traces.append(
                            {
                                **event_params,
                                **trace_params,
                                "starttime": t_start,
                                "remark": "No_waveforms",
                            }
                        )
                        continue
                    if np.all(
                        [
                            x.stats.channel in ["NA", "EA", "UA", "NB", "EB", "UB"]
                            for x in waveforms
                        ]
                    ):
                        waveforms = waveforms.select(channel="[ENU]A").copy()
                        for tr in waveforms:
                            tr.stats.channel = tr.stats.channel.replace("EA", "E")
                            tr.stats.channel = tr.stats.channel.replace("NA", "N")
                            tr.stats.channel = tr.stats.channel.replace("UA", "U")

                    for tr in waveforms:
                        if "UD" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("UD", "Z")
                        if "EW" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("EW", "E")
                        if "NS" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("NS", "N")
                        if "X" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("X", "E")
                        if "Y" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("Y", "N")
                        if "U" in tr.stats.channel:
                            tr.stats.channel = tr.stats.channel.replace("U", "Z")

                    if np.any(
                        ["V" == x.stats.channel[-1:] for x in waveforms]
                    ) and np.all(["Z" != x.stats.channel[-1:] for x in waveforms]):
                        for tr in waveforms:
                            if tr.stats.channel[-1] == "V":
                                tr.stats.channel = tr.stats.channel[:-1] + "Z"

                    waveforms = waveforms.select(channel="[ENZ]").copy()
                    if len(waveforms) == 0:
                        logger.warning(
                            f"Found no waveforms for station {sta} from {t_start} to {t_start+120}"
                        )
                        abnormal_traces.append(
                            {
                                **event_params,
                                **trace_params,
                                "starttime": t_start,
                                "remark": "No_waveforms",
                            }
                        )
                        continue
                    sta_lat = waveforms[0].stats.sac.get("stla", np.nan)
                    sta_lon = waveforms[0].stats.sac.get("stlo", np.nan)
                    sta_elev = waveforms[0].stats.sac.get("stel", np.nan)
                    trace_params.update(
                        {
                            "station_latitude_deg": sta_lat,
                            "station_longitude_deg": sta_lon,
                            "station_elevation_m": sta_elev,
                            "station_epicentral_distance_m": None,
                            "path_azimuth_deg": None,
                            "path_back_azimuth_deg": None,
                        }
                    )
                    # length limit
                    min_starttime = min(trace.stats.starttime for trace in waveforms)
                    max_endtime = max(trace.stats.endtime for trace in waveforms)
                    if max_endtime - min_starttime < win_len_lim:
                        logger.warning(
                            f"Window length is smaller than {win_len_lim} for station {sta} from {t_start} to {t_start+120}"
                        )
                        abnormal_traces.append(
                            {
                                **event_params,
                                **trace_params,
                                "starttime": t_start,
                                "remark": f"length<{win_len_lim}s",
                            }
                        )
                        continue
                    if sampling_rate is None:
                        sampling_rate = waveforms[0].stats.sampling_rate
                    if any(
                        trace.stats.sampling_rate != sampling_rate
                        for trace in waveforms
                    ):
                        logger.warning(
                            f"station {sta} from {t_start} to {t_start+120} | Resampling traces to common sampling rate {sampling_rate}."
                        )
                        waveforms.resample(sampling_rate)
                    trace_params["trace_name"] = (
                        f"""japan_{trace_params['station_code']}{str(min_starttime).replace("-","").replace(":","")[2:-4]}.mseed"""
                    )
                    waveforms.write(
                        download_dir / f"{trace_params['trace_name']}",
                        format="MSEED",
                    )
                    successful_downloads.append(
                        {
                            **event_params,
                            **trace_params,
                            "trace_sampling_rate_hz": sampling_rate,
                        }
                    )
                    logger.info(
                        f"""station {sta} from {t_start} to {t_start+120}| successfully downloaded"""
                    )
                if delete_temp_files:
                    if Path(data.split(".")[0]).is_dir():
                        shutil.rmtree(Path(data.split(".")[0]))
        _save_mseed_log(successful_downloads, log_dir, f"downloads.csv")
        _save_mseed_log(failed_downloads, log_dir, f"failed_downloads.csv")
        _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces.csv")


class NoiseData:
    def __init__(self, save_dir=None, root_folder_name="Noise"):
        self.root_folder_name = root_folder_name
        self._save_dir = save_dir
        self.save_dir = self._save_dir
        self._inventory_path = self.save_dir / "stations.xml"

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if isinstance(value, str):
            value = Path(value)
        self._save_dir = value
        if self._save_dir == None:
            self._save_dir = volpick.cache_root / self.root_folder_name
            print(f"The default saving directory is used: {self._save_dir}")
        else:
            print(f"Set the saving directory to {self._save_dir}")

        try:
            self._save_dir.mkdir(parents=True, exist_ok=False)
            print(f"Create {self._save_dir}")
        except FileExistsError:
            print(f"{self._save_dir} exists")

        # self._save_quakeml_path = self.save_dir / f"{self.cat_file_name}.xml"
        # self._save_csv_path = self.save_dir / f"{self.cat_file_name}.csv"
        self._inventory_path = self.save_dir / "stations.xml"

    def get_inventory(self, client_name="IRIS"):
        """
        Return IRIS inventory
        """
        if not self._inventory_path.exists():  # if the inventory file does not exist
            client = Client(client_name)
            if client_name == "NCEDC":
                kwargs = {}
            else:
                kwargs = {"includerestricted": False}
            inv = client.get_stations(**kwargs)
            print("Downloading the inventory ...")
            inv.write(
                self._inventory_path,
                format="STATIONXML",
            )
        else:
            print(f"{self._inventory_path} exists. Reading the inventory ...")
            inv = read_inventory(self._inventory_path)
        return inv

    def create_noise_table(
        self,
        base_catalog: pd.DataFrame,
        number_stations: int = 200,
        time_difference_limit: float = 3600 * 24,
        number_records_each_station: int = 500,
        output_filename: str = "noise_ref.csv",
    ) -> pd.DataFrame:
        stations = np.unique(
            base_catalog.apply(
                lambda x: (
                    x["station_network_code"],
                    x["station_code"],
                    x["station_location_code"],
                    x["trace_channel"],
                ),
                axis=1,
            ).values
        )
        np.random.seed(100)

        if len(stations) > number_stations:
            np.random.shuffle(stations)
            stations = stations[:number_stations]

        # conservertive_event_end
        def get_event_end(x):
            if pd.notna(x["trace_p_arrival_time"]) and pd.notna(
                x["trace_s_arrival_time"]
            ):
                if x["trace_p_arrival_time"] < x["trace_s_arrival_time"]:
                    return (
                        x["trace_p_arrival_time"]
                        + (x["trace_s_arrival_time"] - x["trace_p_arrival_time"]) * 5
                        + 60
                    )
                else:
                    return x["source_origin_time"] + 600
            elif pd.notna(x["trace_p_arrival_time"]):
                return x["trace_p_arrival_time"] + 600
            else:
                return x["source_origin_time"] + 600

        subcatalogs = []
        for station in stations:
            subcatalog = base_catalog[
                base_catalog.apply(
                    lambda x: (
                        x["station_network_code"],
                        x["station_code"],
                        x["station_location_code"],
                        x["trace_channel"],
                    ),
                    axis=1,
                )
                == station
            ].copy()
            subcatalog["source_origin_time"] = subcatalog["source_origin_time"].apply(
                lambda x: UTCDateTime(x) if pd.notna(x) else np.nan
            )
            subcatalog["trace_p_arrival_time"] = subcatalog[
                "trace_p_arrival_time"
            ].apply(lambda x: UTCDateTime(x) if pd.notna(x) else np.nan)
            subcatalog["trace_s_arrival_time"] = subcatalog[
                "trace_s_arrival_time"
            ].apply(lambda x: UTCDateTime(x) if pd.notna(x) else np.nan)
            subcatalog.sort_values(by=["source_origin_time"], inplace=True)
            subcatalog["event_end"] = subcatalog.apply(get_event_end, axis=1)

            forward_event_time_diff = (
                subcatalog["source_origin_time"].values[1:]
                - subcatalog["event_end"].values[:-1]
            ).astype(float)
            forward_event_time_diff = np.append(forward_event_time_diff, np.nan)
            subcatalog["forward_event_time_difference"] = forward_event_time_diff
            subcatalog["next_event_origin_time"] = np.append(
                subcatalog["source_origin_time"][1:].values, np.nan
            )

            subcatalog.sort_values(
                by=["forward_event_time_difference"], ascending=False, inplace=True
            )
            subcatalog = subcatalog[
                pd.notna(subcatalog["forward_event_time_difference"])
                & (subcatalog["forward_event_time_difference"] > time_difference_limit)
            ]
            subcatalog = subcatalog.iloc[:number_records_each_station].copy()
            subcatalog.reset_index(inplace=True)
            subcatalogs.append(subcatalog)
        result = pd.concat(subcatalogs, ignore_index=True)
        result.to_csv(self.save_dir / output_filename, index=False)
        print(f"Output: {self.save_dir/output_filename}")
        return result

    def download_data(
        self,
        catalog_table: pd.DataFrame,
        time_window: float = 120,
        sampling_rate: float = None,
        num_processes: int = 5,
        download_dir=None,
        win_len_lim=30,
        client_name: str = "IRIS",
        log_level=logging.INFO,
        retry: bool = False,
    ):
        ctx = mp.get_context("spawn")
        mp_start_method = ctx.get_start_method()

        if download_dir is None:
            download_dir = self.save_dir / "mseed"
        if isinstance(download_dir, str):
            download_dir = Path(download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{download_dir} exists")

        # log_dir = self.save_dir / "mseed_log"
        log_dir = download_dir.parent / f"{download_dir.name}_log"
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{log_dir} exists")

        print(
            f"There are {mp.cpu_count()} cpu in this machine. {num_processes} processes are used. Start method: {mp_start_method}"
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
                target=self._download,
                kwargs={
                    "catalog_table": catalog_chunks[i],
                    "time_window": time_window,
                    "sampling_rate": sampling_rate,
                    "download_dir": download_dir,
                    "win_len_lim": win_len_lim,
                    "client_name": client_name,
                    "log_level": log_level,
                },
                name=proc_name,
            )
            process_list.append(proc)
        for i, proc in enumerate(process_list):
            print(
                f"Starting process '{proc.name}'. Chunk size: {len(catalog_chunks[i])}"
            )
            proc.start()
        for proc in process_list:
            proc.join()
            print(f"Finished joining {proc.name}")

        ### Merge csv files generated by subprocesses
        log_files = [
            "downloads.csv",
            "failed_downloads.csv",
            "abnormal_traces.csv",
        ]
        for fname in log_files:
            self._assemble_subprocess_csvlogs(log_dir, fname, retry)

    def _assemble_subprocess_csvlogs(self, log_dir, fname, retry):
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

    def _download(
        self,
        catalog_table,
        time_window: float = 120,
        sampling_rate: float = None,
        download_dir=None,
        win_len_lim=30,
        client_name="IRIS",
        log_level=logging.INFO,
    ):
        # If the process is the main process (sequential downloading),
        # first check and create download_dir and log_dir
        if mp.parent_process() is None:  # check whether it is the main process
            if download_dir is None:
                download_dir = self.save_dir / "mseed"
            if isinstance(download_dir, str):
                download_dir = Path(download_dir)
            try:
                download_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"{download_dir} exists")

        # log_dir = self.save_dir / "mseed_log"
        log_dir = download_dir.parent / f"{download_dir.name}_log"
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
        logger = volpick.logger.getChild("download" + process_mark)
        logger.setLevel(log_level)

        # Initialize an FDSN web service client object
        proc_name = mp.current_process().name
        while True:
            try:
                client = Client(client_name, timeout=600)
            except FDSNNoServiceException:
                logger.error(
                    f"{proc_name}: FDSNNoServiceException. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            except UnicodeDecodeError:
                logger.error(
                    f"{proc_name}: UnicodeDecodeError. Try again. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            except Exception as e:
                # exception_name = type(e).__name__
                logger.error(
                    f"{proc_name}: other error: {type(e).__name__}. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            else:
                break
        inv = self.get_inventory()
        inventory_mapper = InventoryMapper(inv)

        successful_download = []
        abnormal_traces = []
        failed_download = []

        for row in catalog_table.itertuples(name="trace"):
            event_end = UTCDateTime(row.event_end)
            next_event_start = UTCDateTime(row.next_event_origin_time)
            t_start = event_end + 0.5 * (next_event_start - event_end)
            t_end = t_start + time_window
            event_time_difference = row.forward_event_time_difference
            event_params = {
                "source_id": None,
                "source_origin_time": None,
                "source_latitude_deg": None,
                "source_longitude_deg": None,
                "source_depth_km": None,
                "source_magnitude": None,
                "source_magnitude_type": None,
                "source_type": "noise",
            }
            trace_params = {
                "station_network_code": row.station_network_code,
                "station_code": row.station_code,
                "station_location_code": row.station_location_code,
                "trace_channel": row.trace_channel,
                "trace_p_arrival_time": None,
                "trace_s_arrival_time": None,
                "trace_p_max_weight": None,
                "trace_s_max_weight": None,
                "trace_p_first_motion": None,
            }
            waveform_information = (
                f"{trace_params['station_network_code']}.{trace_params['station_code']}"
                f".{trace_params['station_location_code']}.{trace_params['trace_channel']}*, {t_start}---{t_end} |"
            )

            if event_time_difference < 3600:
                abnormal_traces.append(
                    {
                        **event_params,
                        **trace_params,
                        "remark": "short_event_time_difference",
                    }
                )
                continue

            net = row.station_network_code
            sta = row.station_code
            try:
                sta_lat, sta_lon, sta_elev = inventory_mapper.get_station_location(
                    network=net, station=sta
                )
            except KeyError as e:
                logger.warning(f"""{waveform_information} not in inventory""")
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "not_in_inventory"}
                )
                continue
            trace_params.update(
                {
                    "station_latitude_deg": sta_lat,
                    "station_longitude_deg": sta_lon,
                    "station_elevation_m": sta_elev,
                    "station_epicentral_distance_m": None,
                    "path_azimuth_deg": None,
                    "path_back_azimuth_deg": None,
                }
            )

            try:
                waveforms = client.get_waveforms(
                    network=trace_params["station_network_code"],
                    station=trace_params["station_code"],
                    location=trace_params["station_location_code"],
                    channel=f"{trace_params['trace_channel']}*",
                    starttime=t_start,
                    endtime=t_end,
                )
            except Exception as e:
                exception_name = type(e).__name__
                failed_download.append(
                    {**event_params, **trace_params, "Error": exception_name}
                )
                logger.error(f"""{waveform_information} {exception_name}""")
                continue
            rotate_stream_to_zne(waveforms, inv)
            if len(waveforms) == 0:
                logger.warning(
                    f"Found no waveforms for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "No_waveforms"}
                )
                continue
            min_starttime = min(trace.stats.starttime for trace in waveforms)
            max_endtime = max(trace.stats.endtime for trace in waveforms)
            if max_endtime - min_starttime < win_len_lim:
                logger.warning(
                    f"Window length is smaller than {win_len_lim} for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {
                        **event_params,
                        **trace_params,
                        "remark": f"length<{win_len_lim}s",
                    }
                )
                continue
            if sampling_rate is None:
                sampling_rate = waveforms[0].stats.sampling_rate
            if any(trace.stats.sampling_rate != sampling_rate for trace in waveforms):
                logger.warning(
                    f"{waveform_information} Resampling traces to common sampling rate {sampling_rate}."
                )
                waveforms.resample(sampling_rate)
            trace_params["trace_name"] = (
                f"{trace_params['station_network_code']}.{trace_params['station_code']}"
                f""".{trace_params['station_location_code'].replace('--','')}.{trace_params['trace_channel']}{str(min_starttime).replace("-","").replace(":","")[2:-4]}.mseed"""
            )
            waveforms.write(
                download_dir / f"{trace_params['trace_name']}",
                format="MSEED",
            )
            successful_download.append(
                {
                    **event_params,
                    **trace_params,
                    "trace_sampling_rate_hz": sampling_rate,
                }
            )
            logger.info(f"""{waveform_information} successfully downloaded""")

        def _save_mseed_log(records, log_save_dir, fname):
            if len(records) > 0:
                df = pd.DataFrame(data=records)
                df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
                df.to_csv(log_save_dir / fname, index=False)

        _save_mseed_log(successful_download, log_dir, f"downloads{process_mark}.csv")
        _save_mseed_log(failed_download, log_dir, f"failed_downloads{process_mark}.csv")
        _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv")


class AlaskaDataset(CatalogBase):
    """
    Alaska dataset. This class is mainly used for the following task:
    - Read the catalog given in hypoinverse format, and convert it to readable csv format (using
      the seisbench metadata convension).
    - Download waveform data in parallel and saved them as mseed formats. The mseed data are
      downloaded to the "mseed" subdirectory by default. Sucessful downloads, failed downloads
      and abnormal traces are recorded in csv tables in the "mseed_log" subdirectory.

    save_dir: directory to save all related data in. If it is None, 'volpick.cache_root/root_folder_name' will be used.
    root_folder_name: the default name of the folder to store catalog data. If save_dir is set, the parameter `root_folder_name` will take no effect and be ignored.
    cat_file_name: default file where the catalog (including events and arrivals) is written
    """

    def __init__(
        self,
        save_dir=None,
        root_folder_name="Alaska",
        cat_file_name="alaska_catalog",
        **kwargs,
    ):
        self.root_folder_name = root_folder_name
        self.cat_file_name = cat_file_name
        self._save_dir = save_dir
        self.save_dir = self._save_dir
        self._save_quakeml_path = self.save_dir / f"{self.cat_file_name}.xml"
        self._save_csv_path = self.save_dir / f"{self.cat_file_name}.csv"
        self._inventory_path = self.save_dir / "stations.xml"
        super().__init__(**kwargs)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        if isinstance(value, str):
            value = Path(value)
        self._save_dir = value
        if self._save_dir == None:
            self._save_dir = volpick.cache_root / self.root_folder_name
            print(f"The default saving directory is used: {self._save_dir}")
        else:
            print(f"Set the saving directory to {self._save_dir}")

        try:
            self._save_dir.mkdir(parents=True, exist_ok=False)
            print(f"Create {self._save_dir}")
        except FileExistsError:
            print(f"{self._save_dir} exists")

        self._save_quakeml_path = self.save_dir / f"{self.cat_file_name}.xml"
        self._save_csv_path = self.save_dir / f"{self.cat_file_name}.csv"
        self._inventory_path = self.save_dir / "stations.xml"

    def get_inventory(self, client_name="IRIS"):
        """
        Return IRIS inventory
        """
        if not self._inventory_path.exists():  # if the inventory file does not exist
            client = Client(client_name)
            if client_name == "NCEDC":
                kwargs = {}
            else:
                kwargs = {"includerestricted": False}
            inv = client.get_stations(**kwargs)
            print("Downloading the inventory ...")
            inv.write(
                self._inventory_path,
                format="STATIONXML",
            )
        else:
            print(f"{self._inventory_path} exists. Reading the inventory ...")
            inv = read_inventory(self._inventory_path)
        return inv

    def _read_event_summary(self, summary_file, id_prefix):
        summary_dict = {}
        with open(summary_file, "r") as f:
            # skip the first two rows
            f.readline()
            f.readline()
            for line in f:
                event_id = id_prefix + line[99:108].strip()
                event_type = line.strip()[-2:]
                ev_mag = float(line[52:57].strip())
                mag_type = "m" + line[58:60].strip()
                org_time = line[0:25].strip().replace(" ", "T").replace("/", "-")
                lat = float(line[26:34].strip())
                lon = float(line[34:44])
                dep = float(line[44:50])  # km
                summary_dict[event_id] = (
                    event_type,
                    ev_mag,
                    mag_type,
                    org_time,
                    lat,
                    lon,
                    dep,
                )
        return summary_dict

    def _read_an_event(self, f):
        summary_line = f.readline()
        if not summary_line:
            return None, None, None
        station_lines = []
        station_line = f.readline()
        while station_line:
            if not station_line[0:6].strip():
                terminator_line = station_line
                break
            else:
                station_lines.append(station_line)
                station_line = f.readline()
        return summary_line, station_lines, terminator_line

    def read_catalog(
        self,
        station_archive_file,
        summary_file,
        n_events=None,
        id_prefix="",
        min_date=UTCDateTime("1900-01-01T00:00:00.0"),
        max_date=UTCDateTime("2999-01-01T00:00:00.0"),
        save_quakeml=False,
        save_csv=False,
        on_screen=True,
        **kwargs,
    ):
        """
        Read the Alaska catalog given in hypoinverse format

        station_archive_file: a file given in Y2000 (station) archive format (see the hypoinverse document, hy1.40 page 114-115)
        summary_file: a hypoinverse summary file (see the hypoinverse document, hy1.40 page 114-115.
                      The format is a bit different from the document, maybe because of a different version was use)
        n_events: the number of events to be read (useful for testing and debugging)
        id_prefix: a marker string added to the source id
        min_date: only events with origin times larger than 'min_date' will be read
        max_data: only events with origin times less than 'max_date' will be read
        save_quakeml: if true, save the catalog to quakeml format
        save_csv: if true, save the catalog to a csv file. Each column is name by the SeisBench convention
        """
        if n_events is None:
            n_events = np.inf
        summary_dict = self._read_event_summary(
            summary_file=summary_file, id_prefix=id_prefix
        )

        not_in_summary_file = []
        with open(station_archive_file, "r") as f:
            i = 0
            while i < n_events:
                summary_line, station_lines, terminator_line = self._read_an_event(f)
                if not summary_line:  # if it is empty
                    break
                event_id = id_prefix + terminator_line[62:72].strip()
                if summary_line[136:146].strip():
                    assert event_id == id_prefix + summary_line[136:146].strip()

                if event_id not in summary_dict:
                    print(f"Event {event_id} is not in the summary file")
                    not_in_summary_file.append(event_id)
                    continue

                # if the event is in the summary file, perform the following processing

                # information from the summary file
                (
                    event_type,
                    ev_mag,
                    mag_type,
                    org_time_str0,
                    lat0,
                    lon0,
                    dep0,
                ) = summary_dict[event_id]

                # origin time
                if summary_line[0:16].strip():
                    o_year = summary_line[0:4]
                    o_month = summary_line[4:6]
                    o_date = summary_line[6:8]
                    o_hour = summary_line[8:10]
                    o_min = summary_line[10:12]
                    o_sec = summary_line[12:14] + "." + summary_line[14:16]
                    origin_time_str = (
                        o_year
                        + "-"
                        + o_month
                        + "-"
                        + o_date
                        + "T"
                        + o_hour
                        + ":"
                        + o_min
                        + ":"
                        + o_sec
                    )
                else:
                    origin_time_str = org_time_str0
                origin_time = UTCDateTime(origin_time_str)

                if origin_time.timestamp < min_date.timestamp:
                    continue
                if origin_time.timestamp > max_date.timestamp:
                    break

                # hypocenter location
                if summary_line[16:23].strip():  # if the field is not empty
                    hy_lat = (
                        float(summary_line[16:18].strip())
                        + float(summary_line[19:23].strip()) / 100.0 / 60.0
                    )
                    if summary_line[18] == "S":
                        hy_lat = -hy_lat
                else:
                    hy_lat = lat0
                if summary_line[23:31].strip():  # if the field is not empty
                    hy_lon = (
                        float(summary_line[23:26].strip())
                        + float(summary_line[27:31].strip()) / 100 / 60.0
                    )
                    if summary_line[26].isspace():
                        hy_lon = -hy_lon
                else:
                    hy_lon = lon0

                if summary_line[31:36].strip():
                    hy_dep = float(summary_line[31:36].strip()) / 100.0
                else:
                    hy_dep = dep0

                org = obe.Origin()
                org.time = origin_time
                org.latitude = hy_lat  # unit: deg
                org.longitude = hy_lon  # unit: deg
                org.depth = hy_dep  # unit: km

                # horizontal error
                if summary_line[85:89].strip():
                    horizontal_error = float(summary_line[85:89].strip()) / 100.0
                else:
                    horizontal_error = None
                if summary_line[89:93].strip():
                    vertical_error = float(summary_line[89:93].strip()) / 100.0
                else:
                    vertical_error = None
                org_extra = {
                    "horizontal_error": {
                        "value": horizontal_error,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    },
                    "vertical_error": {
                        "value": vertical_error,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    },
                }
                org.extra = org_extra

                # magnitude
                mag = obe.Magnitude(mag=ev_mag, magnitude_type=mag_type)
                mag.origin_id = org.resource_id

                event = obe.Event(resource_id=obe.ResourceIdentifier(id=event_id))
                event.origins.append(org)
                event.preferred_origin_id = event.origins[-1].resource_id
                event.magnitudes.append(mag)
                event.preferred_magnitude_id = event.magnitudes[-1].resource_id
                event_extra = {
                    "source_type": {
                        "value": event_type,
                        "namespace": "http://some-page.de/xmlns/1.0",
                    }
                }
                event.extra = event_extra

                for station_line in station_lines:
                    sta = station_line[0:5].strip()
                    net = station_line[5:7].strip()
                    cha = station_line[9:12].strip()
                    loc = station_line[111:113].strip()
                    try:
                        pick_time_hr_min = UTCDateTime(
                            station_line[17:25] + "T" + station_line[25:29] + "00.0"
                        )
                    except Exception:
                        print(station_archive_file)
                        print(station_line)
                        raise Exception

                    p_remark = station_line[13:15].strip()
                    s_remark = station_line[46:48].strip()
                    p_first_motion = station_line[15].strip()

                    s_pick = None
                    p_pick = None
                    # if not station_line[29:34].isspace():
                    if (not station_line[29:34].isspace()) and len(p_remark) > 0:
                        # if the second is given (not whitespace)
                        p_pick = (
                            pick_time_hr_min
                            + float(station_line[29:34].strip()) / 100.0
                        )

                    # if not station_line[41:46].isspace():
                    if not station_line[41:46].isspace() and len(s_remark) > 0:
                        # if the second is given (not whitespace)
                        s_pick = (
                            pick_time_hr_min
                            + float(station_line[41:46].strip()) / 100.0
                        )

                    if station_line[38:41].strip():
                        pweight = float(station_line[38:41].strip()) / 100.0
                    else:
                        pweight = 0.0
                    if station_line[63:66].strip():
                        sweight = float(station_line[63:66].strip()) / 100.0
                    else:
                        sweight = 0

                    if p_pick:
                        p_pick_obj = obe.Pick(
                            waveform_id=obe.WaveformStreamID(
                                network_code=net,
                                station_code=sta,
                                location_code=loc,
                                channel_code=cha,
                            ),
                            time=p_pick,
                            phase_hint="P",
                        )
                        if p_first_motion == "U":
                            p_pick_obj.polarity = obe.header.PickPolarity.positive
                        elif p_first_motion == "D":
                            p_pick_obj.polarity = obe.header.PickPolarity.negative
                        p_arrival_obj = obe.Arrival(pick_id=p_pick_obj.resource_id)
                        p_arrival_obj.time_weight = pweight
                        event.picks.append(p_pick_obj)
                        event.preferred_origin().arrivals.append(p_arrival_obj)

                    if s_pick:
                        s_pick_obj = obe.Pick(
                            waveform_id=obe.WaveformStreamID(
                                network_code=net,
                                station_code=sta,
                                location_code=loc,
                                channel_code=cha,
                            ),
                            time=s_pick,
                            phase_hint="S",
                        )
                        s_arrival_obj = obe.Arrival(pick_id=s_pick_obj.resource_id)
                        s_arrival_obj.time_weight = sweight
                        event.picks.append(s_pick_obj)
                        event.preferred_origin().arrivals.append(s_arrival_obj)

                    # station_line = f.readline()
                if len(event.picks) > 0:
                    self.events.append(event)
                    i = i + 1

        if save_quakeml:
            print("Writing to quakeml format ...")
            self.write(self._save_quakeml_path, format="QUAKEML")
        if save_csv:
            print("Writing to csv ...")
            self.print(
                by_station=True, on_screen=on_screen, to_file=self._save_csv_path
            )
        elif on_screen:
            self.print(by_station=True, on_screen=on_screen, to_file=False)
        with open(self.save_dir / "not_in_summary_file.txt", "w") as f:
            for sid in not_in_summary_file:
                f.write(sid + "\r\n")

    def read(self, pathname=None, format="csv"):
        """
        Read the whole catalog. If `format` is "quakeml", the catalog is saved to
        the current object (self), and the return value is None.
        If `format` is "csv", the catalog is returned as a pandas DataFrame object.

        pathname: path to the quakeml file or csv file
        format: "quakeml" or "csv"

        return value: None (format="quakeml) or pd.DataFrame (format="csv")
        """
        if format == "quakeml":
            if pathname is None:
                pathname = self._save_quakeml_path
            cat = obe.read_events(pathname)
            self.__dict__.update(cat.copy().__dict__)
            return None
        elif format == "csv":
            if pathname is None:
                pathname = self._save_csv_path
            df = pd.read_csv(pathname)
            return df
        else:
            raise ValueError("The paramerter 'format' must be 'csv' or 'quakeml'")

    def print(self, on_screen=True, by_station=False, to_file=None):
        """
        Show the events and associated picks on the screen or write them into a csv file
        """
        table_items = []
        for event in self.events:
            source_id = str(event.resource_id).split("/")[-1]
            origin = event.preferred_origin()
            mag = event.preferred_magnitude()

            if origin.extra["horizontal_error"]["value"]:
                horizontal_error = float(origin.extra["horizontal_error"]["value"])
            else:
                horizontal_error = None
            if origin.extra["vertical_error"]["value"]:
                vertical_error = float(origin.extra["vertical_error"]["value"])
            else:
                vertical_error = None
            source_type = event.extra["source_type"]["value"]
            if on_screen:
                print(
                    f"""Event {source_id}: {str(origin.time):25s}| {origin.latitude:.4f} (deg), {origin.longitude:.4f} (deg), {origin.depth} (km) | Err(km): h {horizontal_error}, v {vertical_error} | Mag {mag.mag} {mag.magnitude_type} | type: {source_type}"""
                )
            if by_station:
                station_group = defaultdict(list)
                cha_set = []
                for arrival in origin.arrivals:
                    pick_obj = arrival.pick_id.get_referred_object()
                    net, sta, loc, cha = pick_obj.waveform_id.id.split(".")
                    # station_group[f"{net}.{sta}.{loc}.{cha[:-1]}"].append(
                    #     {
                    #         "pick": pick_obj,
                    #         "weight": arrival.time_weight,
                    #     }
                    # )
                    station_group[f"{net}.{sta}.{loc}"].append(
                        {
                            "pick": pick_obj,
                            "weight": arrival.time_weight,
                        }
                    )
                    cha_set.append(f"{net}.{sta}.{loc}.{cha[:-1]}")
                cha_set = list(dict.fromkeys(cha_set))

                average_p_pick_dict = {}
                max_p_weight_dict = {}
                average_s_pick_dict = {}
                max_s_weight_dict = {}
                p_first_motion_dict = {}
                for wave_id, pick_list in station_group.items():
                    net, sta, loc = wave_id.split(".")
                    first_motion_marker = {
                        None: " ",
                        "positive": "U",
                        "negative": "D",
                        "undecidable": " ",
                    }
                    average_p_pick = None
                    max_p_weight = None
                    p_list = []
                    average_s_pick = None
                    max_s_weight = None
                    s_list = []
                    for pick0 in pick_list:
                        if pick0["pick"].phase_hint == "P":
                            p_list.append(pick0)
                        elif pick0["pick"].phase_hint == "S":
                            s_list.append(pick0)

                    p_timestamps = [x["pick"].time.timestamp for x in p_list]
                    p_weights = [x["weight"] for x in p_list]
                    p_first_motion = " "
                    if len(p_timestamps) and sum(p_weights) > 0:
                        average_p_pick = UTCDateTime(
                            np.average(p_timestamps, weights=p_weights)
                        )
                        max_p_weight = max(p_weights)
                        for x in p_list:
                            if x["pick"].polarity:
                                p_first_motion = first_motion_marker[
                                    x["pick"].polarity
                                ]  # use the first record
                                break

                    average_p_pick_dict[wave_id] = average_p_pick
                    max_p_weight_dict[wave_id] = max_p_weight
                    p_first_motion_dict[wave_id] = p_first_motion

                    s_timestamps = [x["pick"].time.timestamp for x in s_list]
                    s_weights = [x["weight"] for x in s_list]

                    if len(s_timestamps) and sum(s_weights) > 0:
                        average_s_pick = UTCDateTime(
                            np.average(s_timestamps, weights=s_weights)
                        )
                        max_s_weight = max(s_weights)
                    average_s_pick_dict[wave_id] = average_s_pick
                    max_s_weight_dict[wave_id] = max_s_weight
                for net_sta_loc_cha in cha_set:
                    net, sta, loc, cha = net_sta_loc_cha.split(".")
                    wave_id = f"{net}.{sta}.{loc}"
                    average_p_pick = average_p_pick_dict[wave_id]
                    average_s_pick = average_s_pick_dict[wave_id]
                    p_first_motion = p_first_motion_dict[wave_id]
                    max_p_weight = max_p_weight_dict[wave_id]
                    max_s_weight = max_s_weight_dict[wave_id]

                    disp_msg = (
                        f"""{sta:5s}{net:5s}{loc:5s}{cha:5s} """
                        f"""| P: {p_first_motion} {average_p_pick}, max weight: {max_p_weight} """
                        f"""| S: {average_s_pick}, max weight: {max_s_weight}"""
                    )

                    if to_file:
                        if len(p_first_motion.strip()) == 0:
                            p_first_motion = None
                        table_items.append(
                            {
                                "source_id": source_id,
                                "source_origin_time": str(origin.time),
                                "source_latitude_deg": origin.latitude,
                                "source_longitude_deg": origin.longitude,
                                "source_depth_km": origin.depth,
                                "source_magnitude": mag.mag,
                                "source_magnitude_type": mag.magnitude_type,
                                "source_type": source_type,
                                "station_network_code": net,
                                "station_code": sta,
                                "station_location_code": loc,
                                "trace_channel": cha,
                                "trace_p_arrival_time": average_p_pick,
                                "trace_s_arrival_time": average_s_pick,
                                "trace_p_max_weight": max_p_weight,
                                "trace_s_max_weight": max_s_weight,
                                "trace_p_first_motion": p_first_motion,
                            }
                        )
                    if on_screen:
                        print(disp_msg)
            else:
                station_group = defaultdict(list)
                for arrival in origin.arrivals:
                    pick_obj = arrival.pick_id.get_referred_object()
                    net, sta, loc, cha = pick_obj.waveform_id.id.split(".")
                    station_group[f"{net}.{sta}.{loc}.{cha}"].append(
                        {
                            "pick": pick_obj,
                            "weight": arrival.time_weight,
                        }
                    )
                for wave_id, pick_list in station_group.items():
                    net, sta, loc, cha = wave_id.split(".")
                    first_motion_marker = {
                        None: " ",
                        "positive": "U",
                        "negative": "D",
                        "undecidable": " ",
                    }
                    disp_msg = f""" {sta:5s} {net:5s} {cha:5s}"""
                    if to_file:
                        table_items.append(
                            {
                                "source_id": source_id,
                                "source_origin_time": str(),
                                "source_latitude_deg": origin.latitude,
                                "source_longitude_deg": origin.longitude,
                                "source_depth_km": origin.depth,
                                "source_magnitude": mag.mag,
                                "source_magnitude_type": mag.magnitude_type,
                                "source_type": source_type,
                                "station_network_code": net,
                                "station_code": sta,
                                "station_location_code": loc,
                                "trace_channel": cha,
                            }
                        )
                    for pick_info_dict in pick_list:
                        pick_obj = pick_info_dict["pick"]
                        pick_weight = pick_info_dict["weight"]
                        disp_msg = (
                            disp_msg
                            + f"""| {pick_obj.phase_hint}: {first_motion_marker[pick_obj.polarity]} {pick_obj.time}, weight: {pick_weight} """
                        )
                        if to_file:
                            table_items[-1].update(
                                {
                                    f"trace_{pick_obj.phase_hint.lower()}_arrival_time": pick_obj.time,
                                    f"trace_{pick_obj.phase_hint.lower()}_weight": pick_weight,
                                }
                            )
                    if on_screen:
                        print(disp_msg)
        if to_file:
            table = pd.DataFrame(table_items)
            table.to_csv(to_file, index=False)

    def retry_failed_downloads(
        self, exclude_list=["FDSNNoDataException"], download_dir=None, **kwargs
    ):
        if download_dir is None:
            download_dir = self.save_dir / "mseed"
        if isinstance(download_dir, str):
            download_dir = Path(download_dir)
        log_dir = download_dir.parent / f"{download_dir.name}_log"
        failed_downloads_log = log_dir / "failed_downloads.csv"
        df = pd.read_csv(failed_downloads_log)
        solvable_errors = df[~df["Error"].isin(exclude_list)].copy()
        if len(solvable_errors) > 0:
            # First remove those error records that are to be dealt with from the log file
            # to avoid possible repeated error records after re-downloading
            excluded_errors = df[df["Error"].isin(exclude_list)].copy()
            excluded_errors.to_csv(failed_downloads_log, index=False, mode="w")

            # Re-download
            self.download_data(
                retry=True,
                catalog_table=solvable_errors,
                download_dir=download_dir,
                **kwargs,
            )

            # Count the number of failed downloads that are still to be resovled
            new_failed_downloads_log = log_dir / "failed_downloads.csv"
            new_df = pd.read_csv(new_failed_downloads_log)
            new_solvable_errors = new_df[~new_df["Error"].isin(exclude_list)].copy()
            return len(new_solvable_errors)
        else:
            print(
                f"All the failed downloads were due to {exclude_list}. They cannot be resolved."
            )
            return len(solvable_errors)

    def download_data(
        self,
        catalog_table: pd.DataFrame,
        time_before: float = 60,
        time_after: float = 60,
        sampling_rate: float = None,
        num_processes: int = 5,
        download_dir=None,
        win_len_lim=5,
        client_name: str = "IRIS",
        log_level=logging.INFO,
        retry: bool = False,
        download_noise: bool = False,
        noise_window_offsets=None,
    ):
        ctx = mp.get_context("spawn")
        mp_start_method = ctx.get_start_method()

        if download_dir is None:
            download_dir = self.save_dir / "mseed"
        if isinstance(download_dir, str):
            download_dir = Path(download_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{download_dir} exists")

        # log_dir = self.save_dir / "mseed_log"
        log_dir = download_dir.parent / f"{download_dir.name}_log"
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{log_dir} exists")

        print(
            f"There are {mp.cpu_count()} cpu in this machine. {num_processes} processes are used. Start method: {mp_start_method}"
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
                target=self._download,
                kwargs={
                    "catalog_table": catalog_chunks[i],
                    "time_before": time_before,
                    "time_after": time_after,
                    "sampling_rate": sampling_rate,
                    "download_dir": download_dir,
                    "win_len_lim": win_len_lim,
                    "client_name": client_name,
                    "log_level": log_level,
                    "download_noise": download_noise,
                    "noise_window_offsets": noise_window_offsets,
                },
                name=proc_name,
            )
            process_list.append(proc)
        for i, proc in enumerate(process_list):
            print(
                f"Starting process '{proc.name}'. Chunk size: {len(catalog_chunks[i])}"
            )
            proc.start()
        for proc in process_list:
            proc.join()
            print(f"Finished joining {proc.name}")

        ### Merge csv files generated by subprocesses
        log_files = [
            "downloads.csv",
            "failed_downloads.csv",
            "abnormal_traces.csv",
        ]
        for fname in log_files:
            self._assemble_subprocess_csvlogs(log_dir, fname, retry)

    def _assemble_subprocess_csvlogs(self, log_dir, fname, retry):
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

    def _download(
        self,
        catalog_table: pd.DataFrame,
        time_before: float = 60,
        time_after: float = 60,
        sampling_rate: float = None,
        download_dir=None,
        win_len_lim=5,
        client_name="IRIS",
        log_level=logging.INFO,
        download_noise=False,
        noise_window_offsets=None,
    ):
        # If the process is the main process (sequential downloading),
        # first check and create download_dir and log_dir
        if mp.parent_process() is None:  # check whether it is the main process
            if download_dir is None:
                download_dir = self.save_dir / "mseed"
            if isinstance(download_dir, str):
                download_dir = Path(download_dir)
            try:
                download_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"{download_dir} exists")

        # log_dir = self.save_dir / "mseed_log"
        log_dir = download_dir.parent / f"{download_dir.name}_log"
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
        logger = volpick.logger.getChild("download" + process_mark)
        logger.setLevel(log_level)

        # Initialize an FDSN web service client object
        proc_name = mp.current_process().name
        while True:
            try:
                client = Client(client_name, timeout=600)
            except FDSNNoServiceException:
                logger.error(
                    f"{proc_name}: FDSNNoServiceException. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            except UnicodeDecodeError:
                logger.error(
                    f"{proc_name}: UnicodeDecodeError. Try again. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            except Exception as e:
                # exception_name = type(e).__name__
                logger.error(
                    f"{proc_name}: other error: {type(e).__name__}. Waiting for 10s before next try ..."
                )
                time.sleep(10)
                continue
            else:
                break

        inv = self.get_inventory()
        inventory_mapper = InventoryMapper(inv)

        successful_download = []
        abnormal_traces = []
        failed_download = []
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
            if download_noise:
                waveform_information = (
                    f"""{event_params["source_id"]}: {event_params["source_origin_time"]} | """
                    f"{trace_params['station_network_code']}.{trace_params['station_code']}"
                    f".{trace_params['station_location_code']}.{trace_params['trace_channel']}* |"
                )

            net = row.station_network_code
            sta = row.station_code
            try:
                sta_lat, sta_lon, sta_elev = inventory_mapper.get_station_location(
                    network=net, station=sta
                )
            except KeyError as e:
                logger.warning(f"""{waveform_information} not in inventory""")
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "not_in_inventory"}
                )
                continue

            if not np.isnan(sta_lat * sta_lon):
                dist, azimuth, back_azimuth = gps2dist_azimuth(
                    event_params["source_latitude_deg"],
                    event_params["source_longitude_deg"],
                    sta_lat,
                    sta_lon,
                )
            else:
                dist = np.nan
                azimuth = np.nan
                back_azimuth = np.nan
            trace_params.update(
                {
                    "station_latitude_deg": sta_lat,
                    "station_longitude_deg": sta_lon,
                    "station_elevation_m": sta_elev,
                    "station_epicentral_distance_m": dist,
                    "path_azimuth_deg": azimuth,
                    "path_back_azimuth_deg": back_azimuth,
                }
            )

            org_time = UTCDateTime(event_params["source_origin_time"])

            if download_noise:
                t_ref = org_time
                noise_window_offsets.sort()
                t_start = t_ref + noise_window_offsets[0]
                t_end = t_ref + noise_window_offsets[1]
            else:
                # quality check
                p_time = None
                s_time = None
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
                    else:
                        t_start = p_time - time_before
                        t_end = s_time + time_after
                elif (p_time is not None) and (s_time is None):
                    if p_time < org_time:
                        abnormal_traces.append(
                            {**event_params, **trace_params, "remark": "P<origin"}
                        )
                        logger.warning(f"""{waveform_information} P<origin""")
                        continue
                    t_start = p_time - time_before
                    t_end = p_time + time_after
                elif (p_time is None) and (s_time is not None):
                    if s_time < org_time:
                        abnormal_traces.append(
                            {**event_params, **trace_params, "remark": "S<origin"}
                        )
                        logger.warning(f"""{waveform_information} S<origin""")
                        continue
                    t_start = s_time - time_before
                    t_end = s_time + time_after

            try:
                waveforms = client.get_waveforms(
                    network=trace_params["station_network_code"],
                    station=trace_params["station_code"],
                    location=trace_params["station_location_code"],
                    channel=f"{trace_params['trace_channel']}*",
                    starttime=t_start,
                    endtime=t_end,
                )

            except Exception as e:
                exception_name = type(e).__name__
                failed_download.append(
                    {**event_params, **trace_params, "Error": exception_name}
                )
                logger.error(f"""{waveform_information} {exception_name}""")
                continue

            rotate_stream_to_zne(waveforms, inv)
            if len(waveforms) == 0:
                logger.warning(
                    f"Found no waveforms for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "No_waveforms"}
                )
                continue

            min_starttime = min(trace.stats.starttime for trace in waveforms)
            max_endtime = max(trace.stats.endtime for trace in waveforms)
            if max_endtime - min_starttime < win_len_lim:
                logger.warning(
                    f"Window length is smaller than {win_len_lim} for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {
                        **event_params,
                        **trace_params,
                        "remark": f"length<{win_len_lim}s",
                    }
                )
                continue

            if not download_noise:
                # If any of the available manual picks is located in a gap, skip it
                arrivals = [xx for xx in [p_time, s_time] if xx is not None]
                check_phases_out_of_traces = [[] for _ in range(len(arrivals))]
                for component in "ZNE":
                    c_stream = waveforms.select(channel=f"*{component}")
                    for arr_i, pha_arr_t in enumerate(arrivals):
                        for trace in c_stream:
                            check_phases_out_of_traces[arr_i].append(
                                pha_arr_t < trace.stats.starttime
                                or pha_arr_t > trace.stats.endtime
                            )
                check_phases_out_of_traces = [
                    np.all(x) for x in check_phases_out_of_traces
                ]
                if np.any(check_phases_out_of_traces):
                    logger.warning(
                        f"P and S are not within the traces for {waveform_information}"
                        f' in event {event_params["source_id"]}'
                    )
                    abnormal_traces.append(
                        {**event_params, **trace_params, "remark": "phases_in_gap"}
                    )
                    continue

            if sampling_rate is None:
                sampling_rate = waveforms[0].stats.sampling_rate

            if any(trace.stats.sampling_rate != sampling_rate for trace in waveforms):
                logger.warning(
                    f"{waveform_information} Resampling traces to common sampling rate {sampling_rate}."
                )
                waveforms.resample(sampling_rate)

            if download_noise:
                event_params.update(
                    {
                        "source_id": f"noise_{row.source_id}",
                        "source_origin_time": None,
                        "source_latitude_deg": None,
                        "source_longitude_deg": None,
                        "source_depth_km": None,
                        "source_magnitude": None,
                        "source_magnitude_type": None,
                        "source_type": "noise",
                    }
                )
                trace_params = {
                    "station_network_code": row.station_network_code,
                    "station_code": row.station_code,
                    "station_location_code": row.station_location_code,
                    "trace_channel": row.trace_channel,
                    "trace_p_arrival_time": None,
                    "trace_s_arrival_time": None,
                    "trace_p_max_weight": None,
                    "trace_s_max_weight": None,
                    "trace_p_first_motion": None,
                    "station_epicentral_distance_m": None,
                    "path_azimuth_deg": None,
                    "path_back_azimuth_deg": None,
                }

            trace_params["trace_name"] = (
                f"{event_params['source_id']}_"
                f"{trace_params['station_network_code']}.{trace_params['station_code']}"
                f".{trace_params['station_location_code'].replace('--','')}.{trace_params['trace_channel']}.mseed"
            )

            waveforms.write(
                download_dir / f"{trace_params['trace_name']}",
                format="MSEED",
            )
            successful_download.append(
                {
                    **event_params,
                    **trace_params,
                    "trace_sampling_rate_hz": sampling_rate,
                }
            )
            logger.info(f"""{waveform_information} successfully downloaded""")

        def _save_mseed_log(records, log_save_dir, fname):
            if len(records) > 0:
                df = pd.DataFrame(data=records)
                df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
                df.to_csv(log_save_dir / fname, index=False)

        _save_mseed_log(successful_download, log_dir, f"downloads{process_mark}.csv")
        _save_mseed_log(failed_download, log_dir, f"failed_downloads{process_mark}.csv")
        _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv")

    def plot_waveforms_with_phases_in_gap(
        self, num=10, dpi=300, waveform_table=None, fig_dir=None, **kwargs
    ):
        if waveform_table is None:
            waveform_table = pd.read_csv(
                self.save_dir / "mseed_log" / "abnormal_traces.csv"
            )
        waveform_table = waveform_table[waveform_table["remark"] == "phases_in_gap"]

        if len(waveform_table) == 0:
            return
        if fig_dir is None:
            fig_dir = self.save_dir / "Phases_out_traces"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=False)
        client = Client("IRIS")
        for i in range(min([len(waveform_table), num])):
            metadata = waveform_table.iloc[i]
            p_time = None
            s_time = None
            if not pd.isna(metadata["trace_p_arrival_time"]):
                p_time = UTCDateTime(metadata["trace_p_arrival_time"])
            if not pd.isna(metadata["trace_s_arrival_time"]):
                s_time = UTCDateTime(metadata["trace_s_arrival_time"])
            arr_times = [x for x in [p_time, s_time] if x is not None]
            t_start = arr_times[0] - 60
            t_end = arr_times[0] + 60
            st = client.get_waveforms(
                network=metadata["station_network_code"],
                station=metadata["station_code"],
                location=metadata["station_location_code"],
                channel=f"{metadata['trace_channel']}*",
                starttime=t_start,
                endtime=t_end,
            )
            starttime = min(trace.stats.starttime for trace in st)
            st.merge()
            nc = len(st)
            fig, axs = plt.subplots(
                nc, 1, figsize=(8, nc * 2.2), sharex="col", squeeze=False
            )
            picks = []
            phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
            for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
                if pd.notna(metadata[phase]):
                    picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

            plt.subplots_adjust(hspace=0.1)
            for k in range(nc):
                axs[k][0].plot(
                    st[k].times(reftime=starttime),
                    st[k].data,
                    label=metadata["source_id"] + "_" + st[k].id,
                    color="black",
                    linewidth=1,
                    **kwargs,
                )
                ymin, ymax = axs[k][0].get_ylim()
                phase_colors = {"P": "blue", "S": "red"}
                for pick, phase_label in picks:
                    axs[k][0].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                    )
                axs[k][0].legend()
            axs[-1][0].set_xlabel(f"Time from {starttime}")
            trace_name = metadata["source_id"] + "_" + st[0].id
            plt.savefig(
                fig_dir / (trace_name + ".jpg"),
                bbox_inches="tight",
                dpi=dpi,
            )

        volpick.logger.info(f"The figures are saved to {fig_dir}")

    def plot_waveforms(
        self,
        indices,
        waveform_table=None,
        data_dir=None,
        fig_dir=None,
        dpi=300,
        savefig=True,
        xrangemin=None,
        xrangemax=None,
        **kwargs,
    ):
        if data_dir == None:
            data_dir = self.save_dir / "mseed"
        else:
            data_dir = Path(data_dir)
        if waveform_table is None:
            waveform_table = pd.read_csv(self.save_dir / "mseed_log" / "downloads.csv")
        if max(indices) > len(waveform_table):
            raise KeyError(
                f"The maximum requested index {max(indices)} is larger than"
                f"the number"
            )
        if fig_dir is None:
            fig_dir = self.save_dir / "mseed_fig"
        volpick.logger.info(f"Plotting {len(indices)} figures")
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=False)
        for i in indices:
            metadata = waveform_table.iloc[i]
            st = read(data_dir / metadata["trace_name"])
            starttime = min(trace.stats.starttime for trace in st)
            st.merge()
            nc = len(st)
            fig, axs = plt.subplots(
                nc, 1, figsize=(8, nc * 2.2), sharex="col", squeeze=False
            )
            picks = []
            phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
            for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
                if pd.notna(metadata[phase]):
                    picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

            plt.subplots_adjust(hspace=0.1)
            for k in range(nc):
                axs[k][0].plot(
                    st[k].times(reftime=starttime),
                    st[k].data,
                    label=st[k].id,
                    color="black",
                    linewidth=1,
                    **kwargs,
                )
                ymin, ymax = axs[k][0].get_ylim()
                phase_colors = {"P": "blue", "S": "red"}
                for pick, phase_label in picks:
                    axs[k][0].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                    )
                axs[k][0].legend()
                if xrangemin is None:
                    xmin = min(st[k].times(reftime=starttime)[0] for k in range(nc))
                else:
                    xmin = min(picks)[0] + xrangemin - starttime
                if xrangemax is None:
                    xmax = max(st[k].times(reftime=starttime)[-1] for k in range(nc))
                else:
                    xmax = max(picks)[0] + xrangemax - starttime
                axs[k][0].set_xlim(xmin, xmax)
            axs[-1][0].set_xlabel(f"Time from {starttime}")
            if savefig:
                plt.savefig(
                    fig_dir / metadata["trace_name"].replace("mseed", "jpg"),
                    bbox_inches="tight",
                    dpi=dpi,
                )
            plt.close()
        volpick.logger.info(f"The figures are saved to {fig_dir}")


class NCEDCDataset(AlaskaDataset):
    def __init__(
        self,
        save_dir=None,
        root_folder_name="ncedc",
        cat_file_name="ncedc_catalog",
        etype="lp",
        **kwargs,
    ):
        super().__init__(
            save_dir=save_dir,
            root_folder_name=root_folder_name,
            cat_file_name=cat_file_name,
            **kwargs,
        )
        self.etype = etype

    def _read_event_summary(self, summary_file, id_prefix):
        summary_dict = {}
        summary_df = pd.read_csv(summary_file, skiprows=1)
        for row in summary_df.itertuples():
            event_id = id_prefix + str(row.EventID)
            # summary_dict[event_id] = (
            #     "lp",
            #     row.Magnitude,
            #     row.MagType,
            # )
            # event_type = "lp"
            event_type = self.etype
            ev_mag = row.Magnitude
            mag_type = row.MagType
            org_time = row.DateTime.replace(" ", "T").replace("/", "-")
            lat = row.Latitude
            lon = row.Longitude
            dep = row.Depth
            summary_dict[event_id] = (
                event_type,
                ev_mag,
                mag_type,
                org_time,
                lat,
                lon,
                dep,
            )
        return summary_dict


class HawaiiDataset(AlaskaDataset):
    def __init__(
        self,
        save_dir=None,
        root_folder_name="hawaii1986to2011",
        cat_file_name="hawaii_catalog1986_2011",
        **kwargs,
    ):
        super().__init__(
            save_dir=save_dir,
            root_folder_name=root_folder_name,
            cat_file_name=cat_file_name,
            **kwargs,
        )

    def _read_event_summary(self, summary_file, id_prefix):
        summary_dict = {}
        with open(summary_file, "r") as f:
            # skip the first two rows
            f.readline()
            f.readline()
            for line in f:
                event_id = id_prefix + line[131:140].strip()
                event_type = line[151:154].strip()
                ev_mag = float(line[117:122].strip())
                try:
                    mag_type = line[124:127].strip()
                except Exception:
                    print(line)
                    print(line[124:127])
                    raise Exception
                if mag_type == "Unk":
                    mag_type = None
                else:
                    mag_type = "m" + mag_type

                org_time = line[0:25].strip().replace(" ", "T").replace("/", "-")
                lat = float(line[26:35].strip())
                lon = float(line[35:46])
                dep = float(line[46:53])  # km
                # summary_dict[event_id] = (event_type, ev_mag, mag_type)
                summary_dict[event_id] = (
                    event_type,
                    ev_mag,
                    mag_type,
                    org_time,
                    lat,
                    lon,
                    dep,
                )
        return summary_dict

    def _read_sac_info(self, fname):
        with open(fname, "r") as f:
            line = f.readline()
            info_dict = {}
            while line:
                key, _, value = line.partition(":")
                info_dict[key] = value.strip().split()
                line = f.readline()
        return info_dict

    def _read_sac_files(self, data_dir, t_offset):
        sac_files = list(data_dir.glob("*.sac"))
        info_files = [Path(str(x).replace("sac", "pick")) for x in sac_files]
        sts = Stream()
        for sac, info in zip(sac_files, info_files):
            st = read(sac)
            info_dict = self._read_sac_info(info)
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

    def convert_sac_to_mseed(
        self,
        catalog_table,
        sampling_rate,
        num_processes=32,
        src_dir=None,
        dest_dir=None,
        win_len_lim=5,
        log_level=logging.INFO,
    ):
        if dest_dir is None:
            dest_dir = self.save_dir / "sac2mseed"
        if isinstance(dest_dir, str):
            dest_dir = Path(dest_dir)
        try:
            dest_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{dest_dir} exists")
        # log_dir = self.save_dir / "sac2mseed_log"
        log_dir = dest_dir.parent / f"{dest_dir.name}_log"
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
                target=self._convert,
                kwargs={
                    "catalog_table": catalog_chunks[i],
                    "sampling_rate": sampling_rate,
                    "src_dir": src_dir,
                    "dest_dir": dest_dir,
                    "win_len_lim": win_len_lim,
                    "log_level": log_level,
                },
                name=proc_name,
            )
            process_list.append(proc)
        for i, proc in enumerate(process_list):
            print(
                f"Starting process '{proc.name}'. Chunk size: {len(catalog_chunks[i])}"
            )
            proc.start()
        for proc in process_list:
            proc.join()
            print(f"Finished joining {proc.name}")
        ### Merge csv files generated by subprocesses
        log_files = [
            "convert.csv",
            "abnormal_traces.csv",
        ]
        for fname in log_files:
            self._assemble_subprocess_csvlogs(log_dir, fname, retry=False)

    def _convert(
        self,
        catalog_table,
        sampling_rate=None,
        src_dir=None,
        dest_dir=None,
        win_len_lim=5,
        log_level=logging.INFO,
    ):
        # If the process is the main process (sequential downloading),
        # first check and create download_dir and log_dir
        if mp.parent_process() is None:  # check whether it is the main process
            if dest_dir is None:
                dest_dir = self.save_dir / "sac2mseed"
            if isinstance(dest_dir, str):
                dest_dir = Path(dest_dir)
            try:
                dest_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"{dest_dir} exists")

        # log_dir = self.save_dir / "sac2mseed_log"
        log_dir = dest_dir.parent / f"{dest_dir.name}_log"
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
        src_dir = Path(src_dir)
        dest_dir = Path(dest_dir)

        # Initialize a logger
        logger = volpick.logger.getChild("convert" + process_mark)
        logger.setLevel(log_level)
        last_subdir = None
        sts = Stream()

        catalog_table.drop_duplicates(
            subset=["source_id", "station_code"], keep="first", inplace=True
        )

        successful_conversion = []
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
                sts = self._read_sac_files(data_dir=subdir, t_offset=36000)

            waveforms = sts.select(station=sta).copy()
            if len(waveforms) == 0:
                logger.warning(f"Found no waveforms for {waveform_information}")
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "No_waveforms"}
                )
                continue

            if np.all([x.stats.channel == "" for x in waveforms]):
                waveforms = waveforms[:1]
                waveforms[0].stats.channel = "Z"
            waveforms = waveforms.select(channel="*[ENZV]").copy()

            if len(waveforms) == 0:
                logger.warning(f"Found no waveforms for {waveform_information}")
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "No_waveforms"}
                )
                continue

            if np.any(["V" == x.stats.channel[-1:] for x in waveforms]) and np.any(
                ["Z" == x.stats.channel[-1:] for x in waveforms]
            ):
                waveforms = waveforms.select(channel="*[ENV]").copy()

            for tr in waveforms:
                if len(tr.stats.channel) == 1:
                    tr.stats.channel = trace_params["trace_channel"] + tr.stats.channel
                if tr.stats.channel[-1] == "V":
                    tr.stats.channel = tr.stats.channel[:-1] + "Z"

            sta_lat = waveforms[0].stats.sac.get("stla", np.nan)
            sta_lon = waveforms[0].stats.sac.get("stlo", np.nan)
            sta_elev = waveforms[0].stats.sac.get("stel", np.nan)

            if not np.isnan(sta_lat * sta_lon):
                dist, azimuth, back_azimuth = gps2dist_azimuth(
                    event_params["source_latitude_deg"],
                    event_params["source_longitude_deg"],
                    sta_lat,
                    sta_lon,
                )
            else:
                dist = np.nan
                azimuth = np.nan
                back_azimuth = np.nan
            trace_params.update(
                {
                    "station_latitude_deg": sta_lat,
                    "station_longitude_deg": sta_lon,
                    "station_elevation_m": sta_elev,
                    "station_epicentral_distance_m": dist,
                    "path_azimuth_deg": azimuth,
                    "path_back_azimuth_deg": back_azimuth,
                }
            )

            # length limit
            min_starttime = min(trace.stats.starttime for trace in waveforms)
            max_endtime = max(trace.stats.endtime for trace in waveforms)
            if max_endtime - min_starttime < win_len_lim:
                logger.warning(
                    f"Window length is smaller than {win_len_lim} for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {
                        **event_params,
                        **trace_params,
                        "remark": f"length<{win_len_lim}s",
                    }
                )
                continue

            # If any of the available manual picks is located in a gap, skip it
            arrivals = [xx for xx in [p_time, s_time] if xx is not None]
            check_phases_out_of_traces = [[] for _ in range(len(arrivals))]
            for component in "ZNE":
                c_stream = waveforms.select(channel=f"*{component}")
                for arr_i, pha_arr_t in enumerate(arrivals):
                    for trace in c_stream:
                        check_phases_out_of_traces[arr_i].append(
                            pha_arr_t < trace.stats.starttime
                            or pha_arr_t > trace.stats.endtime
                        )
            check_phases_out_of_traces = [np.all(x) for x in check_phases_out_of_traces]
            if np.any(check_phases_out_of_traces):
                logger.warning(
                    f"P and S are not within the traces for {waveform_information}"
                    f' in event {event_params["source_id"]}'
                )
                abnormal_traces.append(
                    {**event_params, **trace_params, "remark": "phases_in_gap"}
                )
                continue

            if sampling_rate is None:
                sampling_rate = waveforms[0].stats.sampling_rate

            if any(trace.stats.sampling_rate != sampling_rate for trace in waveforms):
                logger.warning(
                    f"{waveform_information} Resampling traces to common sampling rate {sampling_rate}."
                )
                waveforms.resample(sampling_rate)

            trace_params["trace_name"] = (
                f"{event_params['source_id']}_"
                f"{trace_params['station_network_code']}.{trace_params['station_code']}"
                f".{trace_params['station_location_code'].replace('--','')}.{trace_params['trace_channel']}.mseed"
            )
            successful_conversion.append(
                {
                    **event_params,
                    **trace_params,
                    "trace_sampling_rate_hz": sampling_rate,
                }
            )
            waveforms.write(
                dest_dir / f"{trace_params['trace_name']}",
                format="MSEED",
            )
            logger.info(f"""{waveform_information} successfully converted to mseed""")

        def _save_mseed_log(records, log_save_dir, fname):
            if len(records) > 0:
                df = pd.DataFrame(data=records)
                df.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
                df.to_csv(log_save_dir / fname, index=False)

        _save_mseed_log(successful_conversion, log_dir, f"convert{process_mark}.csv")
        _save_mseed_log(abnormal_traces, log_dir, f"abnormal_traces{process_mark}.csv")

    def plot_sac_waveforms_with_phases_in_gap(
        self, num=10, dpi=300, src_dir=None, waveform_table=None, fig_dir=None, **kwargs
    ):
        src_dir = Path(src_dir)
        if waveform_table is None:
            waveform_table = pd.read_csv(
                self.save_dir / "mseed_log" / "abnormal_traces.csv"
            )
        waveform_table = waveform_table[waveform_table["remark"] == "phases_in_gap"]
        print(len(waveform_table))
        if len(waveform_table) == 0:
            return
        if fig_dir is None:
            fig_dir = self.save_dir / "Phases_out_traces"
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=False)
        subdir = None
        last_subdir = None
        sts = Stream()
        for i in range(min([len(waveform_table), num])):
            metadata = waveform_table.iloc[i]
            p_time = None
            s_time = None
            if not pd.isna(metadata["trace_p_arrival_time"]):
                p_time = UTCDateTime(metadata["trace_p_arrival_time"])
            if not pd.isna(metadata["trace_s_arrival_time"]):
                s_time = UTCDateTime(metadata["trace_s_arrival_time"])

            sta = metadata["station_code"]
            year = metadata["source_origin_time"].split("T")[0].split("-")[0]
            month = metadata["source_origin_time"].split("T")[0].split("-")[1]
            evid = metadata["source_id"].replace("hawaii", "")

            subdir = src_dir / year / month / f"{evid}.dir"
            if subdir != last_subdir:
                last_subdir = subdir
                sts = self._read_sac_files(data_dir=subdir, t_offset=36000)

            st = sts.select(station=sta).copy()
            # st = st.select(channel="*[ENZV]").copy()

            starttime = min(trace.stats.starttime for trace in st)
            # print(len(st))
            st.merge()
            nc = len(st)
            fig, axs = plt.subplots(
                nc, 1, figsize=(8, nc * 2.2), sharex="col", squeeze=False
            )
            picks = []
            phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
            for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
                if pd.notna(metadata[phase]):
                    picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

            plt.subplots_adjust(hspace=0.1)
            for k in range(nc):
                axs[k][0].plot(
                    st[k].times(reftime=starttime),
                    st[k].data,
                    label=metadata["source_id"] + "_" + st[k].id,
                    color="black",
                    linewidth=1,
                    **kwargs,
                )
                ymin, ymax = axs[k][0].get_ylim()
                phase_colors = {"P": "blue", "S": "red"}
                for pick, phase_label in picks:
                    axs[k][0].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                    )
                axs[k][0].legend()
            axs[-1][0].set_xlabel(f"Time from {starttime}")
            trace_name = metadata["source_id"] + "_" + st[0].id
            plt.savefig(
                fig_dir / (trace_name + ".jpg"),
                bbox_inches="tight",
                dpi=dpi,
            )
            plt.close()

        volpick.logger.info(f"The figures are saved to {fig_dir}")


class ComCatDataset(AlaskaDataset):
    def __init__(
        self,
        save_dir=None,
        root_folder_name="comcat",
        cat_file_name="phases",
        **kwargs,
    ):
        super().__init__(
            save_dir=save_dir,
            root_folder_name=root_folder_name,
            cat_file_name=cat_file_name,
            **kwargs,
        )

    def group_picks(self, phase):
        station_group = {}
        for j in range(len(phase)):
            row = phase.iloc[j]
            net, sta, cha, loc = row["Channel"].split(".")
            if len(loc.strip()) == 0:
                loc = "--"
            # print(net,sta,cha,temp)
            if f"{net}.{sta}.{cha[:-1]}" not in station_group:
                station_group[f"{net}.{sta}.{cha[:-1]}"] = {
                    "p_picks": [],
                    "p_weights": [],
                    "s_picks": [],
                    "s_weights": [],
                    "station_network_code": net,
                    "station_code": sta,
                    "trace_channel": cha[:-1],
                    "station_location_code": loc,
                    "trace_p_status": None,
                    "trace_s_status": None,
                }
            pha_type = row["Phase"].lower()
            try:
                station_group[f"{net}.{sta}.{cha[:-1]}"][f"{pha_type}_picks"].append(
                    row["Arrival Time"].timestamp()
                )
            except KeyError as e:
                print(phase)
                raise e
            station_group[f"{net}.{sta}.{cha[:-1]}"][f"{pha_type}_weights"].append(
                row["Weight"]
            )
            station_group[f"{net}.{sta}.{cha[:-1]}"][f"trace_{pha_type}_status"] = row[
                "Status"
            ]

        for sta_id in station_group.keys():
            for pha in ["p", "s"]:
                station_group[sta_id][f"trace_{pha}_first_motion"] = None
                picks_ts = station_group[sta_id][f"{pha}_picks"]
                current_weights = station_group[sta_id][f"{pha}_weights"]
                if len(picks_ts) > 0:
                    try:
                        station_group[sta_id][f"trace_{pha}_arrival_time"] = (
                            UTCDateTime(
                                np.average(
                                    picks_ts,
                                    weights=current_weights,
                                )
                            )
                        )
                    except ZeroDivisionError as e1:
                        if np.allclose(current_weights, 0):
                            station_group[sta_id][f"trace_{pha}_arrival_time"] = (
                                UTCDateTime(np.average(picks_ts))
                            )
                        else:
                            raise e1
                    except TypeError:
                        # if all(np.isnan(current_weights)):
                        #     station_group[sta_id][f"trace_{pha}_arrival_time"] = (
                        #         UTCDateTime(np.average(picks_ts))
                        #     )
                        if not all(np.isnan(current_weights)):
                            if np.nanmax(current_weights) > 0:
                                picks_ts = [
                                    x
                                    for x, y in zip(
                                        picks_ts,
                                        current_weights,
                                    )
                                    if not np.isnan(y)
                                ]
                        station_group[sta_id][f"trace_{pha}_arrival_time"] = (
                            UTCDateTime(np.average(picks_ts))
                        )
                    if np.all(np.isnan(current_weights)):
                        station_group[sta_id][f"trace_{pha}_max_weight"] = np.nan
                    else:
                        station_group[sta_id][f"trace_{pha}_max_weight"] = np.nanmax(
                            current_weights
                        )
                else:
                    station_group[sta_id][f"trace_{pha}_arrival_time"] = None
                    station_group[sta_id][f"trace_{pha}_max_weight"] = None

        return station_group

    def download_phases(self, summary_df):
        pick_df = []
        events_without_picks = []
        events_without_picks_row_idx_list = []
        for i in tqdm(range(len(summary_df))):
            current_row = summary_df.iloc[i]
            source_id = current_row["id"]
            try:
                detail = get_event_by_id(source_id, includesuperseded=True)
            except JSONDecodeError:
                print(f"Cannot find the event {source_id}")
                events_without_picks_row_idx_list.append(i)
                continue
            detail_dict = detail.toDict()
            source_params = {
                "source_id": source_id,
                "source_origin_time": UTCDateTime(current_row["time"]),
                "source_latitude_deg": current_row["latitude"],
                "source_longitude_deg": current_row["longitude"],
                "source_depth_km": current_row["depth"],
                "source_magnitude": detail_dict["magnitude"],
                "source_magnitude_type": detail_dict["magtype"],
                "source_type": current_row["eventtype"],
            }
            try:
                phase = None
                while phase is None:
                    phase = get_phase_dataframe(detail)
            except KeyError:
                print(f"Arrival time is not available for: {detail.id}")
                events_without_picks.append(detail.id)
                events_without_picks_row_idx_list.append(i)
                continue
            phase.replace({"Pn": "P", "Pg": "P", "Sn": "S", "Sg": "S"}, inplace=True)
            phase.sort_values(by=["Channel"], inplace=True)
            station_group = self.group_picks(phase)
            for sta_key in station_group:
                current_sta = station_group[sta_key]
                trace_params = {
                    "station_network_code": current_sta["station_network_code"],
                    "station_code": current_sta["station_code"],
                    "station_location_code": current_sta["station_location_code"],
                    "trace_channel": current_sta["trace_channel"],
                    "trace_p_arrival_time": current_sta["trace_p_arrival_time"],
                    "trace_s_arrival_time": current_sta["trace_s_arrival_time"],
                    "trace_p_max_weight": current_sta["trace_p_max_weight"],
                    "trace_s_max_weight": current_sta["trace_s_max_weight"],
                    "trace_p_status": current_sta["trace_p_status"],
                    "trace_s_status": current_sta["trace_s_status"],
                    "trace_p_first_motion": current_sta["trace_p_first_motion"],
                    "trace_s_first_motion": current_sta["trace_s_first_motion"],
                }
                pick_df.append({**source_params, **trace_params})
        pick_df = pd.DataFrame(data=pick_df)
        pick_df.to_csv(self._save_csv_path, index=False)
        summary_df.iloc[events_without_picks_row_idx_list].to_csv(
            self.save_dir / "events_without_picks.csv", index=False
        )

    def read_PNSN_events(self, pnsn_events_export_filename, source_type):
        summary_df = pd.read_csv(pnsn_events_export_filename)
        summary_df["eventtype"] = source_type
        summary_df.rename(
            columns={
                "Time UTC": "time",
                "Evid": "id",
                "Lat": "latitude",
                "Lon": "longitude",
                "Depth Km": "depth",
                "Magnitude": "magnitude",
                "Magnitude Type": "magtype",
            },
            inplace=True,
        )
        summary_df["id"] = summary_df["id"].apply(lambda x: f"uw{x}")
        return summary_df


if __name__ == "__main__":
    ncedc = NCEDCDataset()
