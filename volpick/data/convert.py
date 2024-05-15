from typing import Union
import numpy as np
import os
import pandas as pd

from obspy import UTCDateTime
from obspy import read
from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth

import seisbench
import seisbench.data as sbd
from seisbench.util.trace_ops import (
    rotate_stream_to_zne,
    trace_has_spikes,
    waveform_id_to_network_station_location,
)

from pathlib import Path
from .utils import freqency_index, calculate_snr
import volpick


# This function is adapted from seisbench.util.trace_ops. The only change is
# c_stream = stream.select(channel=f"??{c}") --> c_stream = stream.select(channel=f"*{c}")
def stream_to_array(stream, component_order):
    """
    Converts stream of single station waveforms into a numpy array according to a given component order.
    If trace start and end times disagree between component traces, remaining parts are filled with zeros.
    Also returns completeness, i.e., the fraction of samples in the output that actually contain data.
    Assumes all traces to have the same sampling rate.

    :param stream: Stream to convert
    :type stream: obspy.Stream
    :param component_order: Component order
    :type component_order: str
    :return: starttime, data, completeness
    :rtype: UTCDateTime, np.ndarray, float
    """
    starttime = min(trace.stats.starttime for trace in stream)
    endtime = max(trace.stats.endtime for trace in stream)
    sampling_rate = stream[0].stats.sampling_rate

    samples = int((endtime - starttime) * sampling_rate) + 1

    completeness = 0.0
    data = np.zeros((len(component_order), samples), dtype="float64")
    for c_idx, c in enumerate(component_order):
        c_stream = stream.select(channel=f"*{c}")
        if len(c_stream) > 1:
            # If multiple traces are found, issue a warning and write them into the data ordered by their length
            volpick.logger.warning(
                f"Found multiple traces for {c_stream[0].id} starting at {stream[0].stats.starttime}. "
                f"Completeness will be wrong in case of overlapping traces."
            )
            c_stream = sorted(c_stream, key=lambda x: x.stats.npts)

        c_completeness = 0.0
        for trace in c_stream:
            start_sample = int((trace.stats.starttime - starttime) * sampling_rate)
            l = min(len(trace.data), samples - start_sample)
            data[c_idx, start_sample : start_sample + l] = trace.data[:l]
            c_completeness += l

        completeness += min(1.0, c_completeness / samples)

    data -= np.mean(data, axis=1, keepdims=True)

    completeness /= len(component_order)
    return starttime, data, completeness


def convert_mseed_to_seisbench(
    catalog_table: pd.DataFrame,
    mseed_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    split_prob=[0.75, 0.1, 0.15],
    chunk="",
    is_japan_format=False,
    check_long_traces=False,
    check_long_traces_limit=150,
    skip_spikes=False,
    cut_bounds=None,
    n_limit=None,
):
    """
    Convert data into seisbench format
    # data is demeaned before converting to seisbench format
    """
    dest_dir = Path(dest_dir)
    mseed_dir = Path(mseed_dir)
    metadata_path = dest_dir / f"metadata{chunk}.csv"
    waveforms_path = dest_dir / f"waveforms{chunk}.hdf5"
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        writer.data_format = {  # Define data format
            "dimension_order": "CW",
            "component_order": "ZNE",
            # "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        n_write = 0
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
            if not is_japan_format:
                trace_params = {
                    "station_network_code": row.station_network_code,
                    "station_code": row.station_code,
                    "station_location_code": row.station_location_code,
                    "trace_channel": row.trace_channel,
                    "station_latitude_deg": row.station_latitude_deg,
                    "station_longitude_deg": row.station_longitude_deg,
                    "station_elevation_m": row.station_elevation_m,
                    "station_epicentral_distance_m": row.station_epicentral_distance_m,
                    "path_azimuth_deg": row.path_azimuth_deg,
                    "path_back_azimuth_deg": row.path_back_azimuth_deg,
                    "trace_p_arrival_time": row.trace_p_arrival_time,
                    "trace_s_arrival_time": row.trace_s_arrival_time,
                    "trace_p_max_weight": row.trace_p_max_weight,
                    "trace_s_max_weight": row.trace_s_max_weight,
                    "trace_p_first_motion": row.trace_p_first_motion,
                    "trace_name": row.trace_name,
                }
            else:
                trace_params = {
                    "station_network_code": row.station_network_code,
                    "station_code": row.station_code,
                    "station_location_code": row.station_location_code,
                    "trace_channel": row.trace_channel,
                    "station_latitude_deg": row.station_latitude_deg,
                    "station_longitude_deg": row.station_longitude_deg,
                    "station_elevation_m": row.station_elevation_m,
                    "station_epicentral_distance_m": row.station_epicentral_distance_m,
                    "path_azimuth_deg": row.path_azimuth_deg,
                    "path_back_azimuth_deg": row.path_back_azimuth_deg,
                    "trace_p_arrival_time": row.trace_p_arrival_time,
                    "trace_s_arrival_time": row.trace_s_arrival_time,
                    "trace_name": row.trace_name,
                }

            waveforms = read(mseed_dir / trace_params["trace_name"])  # read miniseed

            sampling_rate = 100  # waveforms[0].stats.sampling_rate
            if any(trace.stats.sampling_rate != sampling_rate for trace in waveforms):
                volpick.logger.warning(
                    f"""Found inconsistent sampling rates for """
                    f"""{waveform_id_to_network_station_location(waveforms[0].id)} in event {event_params["source_id"]}."""
                    f"""Resampling traces to common sampling rate."""
                )
                waveforms.resample(sampling_rate)
            trace_params["trace_sampling_rate_hz"] = sampling_rate

            waveforms.detrend("demean")

            min_starttime = min(trace.stats.starttime for trace in waveforms)
            max_endtime = max(trace.stats.endtime for trace in waveforms)
            # print(isinstance(cut_bounds, (int, float)))
            if isinstance(cut_bounds, (int, float)):
                # print("Cut bounds")
                if (max_endtime - min_starttime) > (3 * cut_bounds + 60):
                    waveforms.trim(
                        starttime=min_starttime + cut_bounds,
                        endtime=max_endtime - cut_bounds,
                    )

            min_starttime = min(trace.stats.starttime for trace in waveforms)
            max_endtime = max(trace.stats.endtime for trace in waveforms)
            if check_long_traces & (
                (max_endtime - min_starttime) > check_long_traces_limit
            ):
                arr_times = []
                for arr_time in [
                    trace_params["trace_p_arrival_time"],
                    trace_params["trace_s_arrival_time"],
                ]:
                    if pd.notna(arr_time):
                        arr_times.append(UTCDateTime(arr_time))

                waveforms.trim(
                    starttime=max(
                        min(arr_times) - check_long_traces_limit / 2, min_starttime
                    ),
                    endtime=min(
                        max(arr_times) + check_long_traces_limit / 2, max_endtime
                    ),
                )

            actual_t_start, data, completeness = stream_to_array(
                waveforms,
                component_order=writer.data_format["component_order"],
            )

            arrival_times = {
                "p": row.trace_p_arrival_time,
                "s": row.trace_s_arrival_time,
            }
            trace_params["trace_has_spikes"] = trace_has_spikes(data)
            if skip_spikes:
                if trace_params["trace_has_spikes"]:
                    continue
            trace_params["trace_start_time"] = str(actual_t_start)
            for phase_hint, phase_arrival_time in arrival_times.items():
                if not pd.isna(phase_arrival_time):
                    sample = (
                        UTCDateTime(phase_arrival_time) - actual_t_start
                    ) * sampling_rate
                    trace_params[f"trace_{phase_hint}_arrival_sample"] = int(sample)
                    trace_params[f"trace_{phase_hint}_status"] = "None"
                else:
                    trace_params[f"trace_{phase_hint}_arrival_sample"] = None
                    trace_params[f"trace_{phase_hint}_status"] = None

            # snr
            # if (trace_params[f"trace_p_arrival_sample"] is not None) and (trace_params[f"trace_s_arrival_sample"] is not None):
            snrs, average_snr = calculate_snr(
                data,
                trace_params["trace_p_arrival_sample"],
                trace_params["trace_s_arrival_sample"],
                5 * trace_params["trace_sampling_rate_hz"],
            )
            trace_params["trace_snr_db"] = snrs
            trace_params["trace_mean_snr_db"] = average_snr

            if pd.notna(trace_params["trace_p_arrival_sample"]) or pd.notna(
                trace_params["trace_s_arrival_sample"]
            ):
                # add frequency index
                win_before = 1 * sampling_rate
                win_after = 6 * sampling_rate
                component_FIs = []
                if trace_params["trace_p_arrival_sample"]:
                    ref_sample = trace_params["trace_p_arrival_sample"]
                elif trace_params["trace_s_arrival_sample"]:
                    ref_sample = trace_params["trace_s_arrival_sample"]
                else:
                    ref_sample = None
                if ref_sample:
                    for a_component in data:
                        if np.sum(np.abs(np.diff(a_component))) > 1e-9:
                            try:
                                fi = freqency_index(
                                    data=a_component[
                                        max(ref_sample - win_before, 0) : min(
                                            ref_sample + win_after, len(a_component)
                                        )
                                    ],
                                    dt=1.0 / sampling_rate,
                                    low_freq_band=[1, 5],
                                    high_freq_band=[10, 15],
                                )
                            except Exception:
                                print(trace_params["trace_name"])
                                raise Exception

                            if not np.isnan(fi):
                                component_FIs.append(fi)
                if len(component_FIs) > 0:
                    trace_FI = np.mean(component_FIs)
                trace_params["trace_frequency_index"] = trace_FI
            else:
                trace_params["trace_frequency_index"] = np.nan

            # split
            trace_params["split"] = np.random.choice(
                ["train", "dev", "test"], p=split_prob
            )
            writer.add_trace({**event_params, **trace_params}, data)
            if n_limit is not None:
                n_write = n_write + 1
                if n_write >= n_limit:
                    break
    metadata = pd.read_csv(metadata_path)
    if np.all(
        pd.notna(metadata["trace_p_arrival_sample"])
        | pd.notna(metadata["trace_s_arrival_sample"])
    ):
        source_ids = np.unique(list(metadata["source_id"]))
        source_fi_dict = {}
        for sid in source_ids:
            source_fi_dict[sid] = np.mean(
                metadata[metadata["source_id"] == sid]["trace_frequency_index"]
            )
        source_frequency_index = [
            source_fi_dict[x] for x in metadata["source_id"].to_numpy()
        ]
        metadata["source_frequency_index"] = source_frequency_index
    else:
        metadata["source_frequency_index"] = np.nan
    metadata.to_csv(metadata_path, index=False)

    # trace_frequency_index, trace_p_arrival_sample, trace_s_arrival_sample, trace_p_status, trace_s_status, trace_start_time, trace_has_spikes


# Deprecated
# I used to save waveforms and metadata of different events to to different folders.
# This way of data management was very inefficient and inconvenient.
def convert_from_old_format(
    src_dir, dest_dir, bucket_size=1024, split_prob=[0.7, 0.1, 0.2]
):
    def get_event_params(event):
        event_df = pd.read_csv(event / "event_info.csv", index_col=0)
        event_info = event_df.iloc[0]
        event_params = {
            "source_origin_time": event_info["origin_time"],
            "source_id": event_info["event_id"],
            "source_origin_time": event_info["origin_time"],
            # "source_origin_uncertainty_sec": origin.time_errors["uncertainty"],
            "source_latitude_deg": event_info["hypo_lat"],
            # "source_latitude_uncertainty_km": origin.latitude_errors["uncertainty"],
            "source_longitude_deg": event_info["hypo_lon"],
            # "source_longitude_uncertainty_km": origin.longitude_errors["uncertainty"],
            "source_depth_km": event_info["hypo_depth"],
            # "source_depth_uncertainty_km": origin.depth_errors["uncertainty"] / 1e3,
            "source_magnitude": event_info["magnitude"],
            "source_type": event_info["event_type"],
        }
        return event_params

    def get_trace_params(pick, event_params):
        lat = pick["latitude"]
        lon = pick["longitude"]
        if not np.isnan(lat * lon):
            back_azimuth = gps2dist_azimuth(
                event_params["source_latitude_deg"],
                event_params["source_longitude_deg"],
                lat,
                lon,
            )[2]
        else:
            back_azimuth = np.nan
        trace_params = {
            "station_network_code": pick["network"],
            "station_code": pick["station"],
            "trace_channel": pick["instrument"],
            "station_location_code": None,
            "station_latitude_deg": pick["latitude"],
            "station_longitude_deg": pick["longitude"],
            "station_elevation_m": pick["elevation_m"],
            "path_back_azimuth_deg": back_azimuth,
        }
        return trace_params

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    event_dirs = sorted([x for x in src_dir.iterdir() if x.is_dir()])
    metadata_path = dest_dir / "metadata.csv"
    waveforms_path = dest_dir / "waveforms.hdf5"

    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        writer.data_format = {  # Define data format
            "dimension_order": "CW",
            "component_order": "ZNE",
            # "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        writer.bucket_size = bucket_size
        for event_dir in event_dirs:
            event_params = get_event_params(event_dir)
            pickdf = pd.read_csv(event_dir / "picks.csv", index_col=0)
            for i in range(len(pickdf)):
                pick = pickdf.iloc[i]
                trace_params = get_trace_params(pick, event_params)
                waveforms = read(event_dir / pick.name)  # read miniseed
                inv = read_inventory(event_dir / pick.name.replace("mseed", "xml"))

                rotate_stream_to_zne(waveforms, inv)

                sampling_rate = 100  # waveforms[0].stats.sampling_rate
                if any(
                    trace.stats.sampling_rate != sampling_rate for trace in waveforms
                ):
                    seisbench.logger.warning(
                        f"Found inconsistent sampling rates for "
                        f"{waveform_id_to_network_station_location(waveforms[0].id)} in event {event_dir.name}."
                        f"Resampling traces to common sampling rate."
                    )
                    waveforms.resample(sampling_rate)
                trace_params["trace_sampling_rate_hz"] = sampling_rate
                trace_params["trace_name"] = (
                    f"{event_params['source_id']}_{waveform_id_to_network_station_location(waveforms[0].id)}"
                )

                actual_t_start, data, completeness = stream_to_array(
                    waveforms,
                    component_order=writer.data_format["component_order"],
                )

                trace_params["trace_has_spikes"] = trace_has_spikes(data)
                trace_params["trace_start_time"] = str(actual_t_start)
                # picks
                for phase_hint in ["p", "s"]:
                    if not pd.isna(pick[f"{phase_hint.lower()}_time"]):
                        sample = (
                            UTCDateTime(pick[f"{phase_hint.lower()}_time"])
                            - actual_t_start
                        ) * sampling_rate
                        trace_params[f"trace_{phase_hint}_arrival_sample"] = int(sample)
                        trace_params[f"trace_{phase_hint}_status"] = "USGS"
                    else:
                        trace_params[f"trace_{phase_hint}_arrival_sample"] = None
                        trace_params[f"trace_{phase_hint}_status"] = None
                trace_params[f"trace_p_first_motion"] = pick["first_motion"]

                # add frequency index
                win_before = 1 * sampling_rate
                win_after = 6 * sampling_rate
                component_FIs = []
                if trace_params["trace_p_arrival_sample"]:
                    ref_sample = trace_params["trace_p_arrival_sample"]
                elif trace_params["trace_s_arrival_sample"]:
                    ref_sample = trace_params["trace_s_arrival_sample"]
                else:
                    ref_sample = None
                if ref_sample:
                    for a_component in data:
                        if np.sum(np.abs(np.diff(a_component))) > 1e-9:
                            fi = freqency_index(
                                data=a_component[
                                    ref_sample - win_before : ref_sample + win_after
                                ],
                                dt=1.0 / sampling_rate,
                                low_freq_band=[1, 5],
                                high_freq_band=[10, 15],
                            )
                            if not np.isnan(fi):
                                component_FIs.append(fi)
                if len(component_FIs) > 0:
                    trace_FI = np.mean(component_FIs)
                trace_params["trace_frequency_index"] = trace_FI

                # split
                trace_params["split"] = np.random.choice(
                    ["train", "dev", "test"], p=split_prob
                )
                writer.add_trace({**event_params, **trace_params}, data)

    metadata = pd.read_csv(metadata_path)
    source_ids = np.unique(list(metadata["source_id"]))
    source_fi_dict = {}
    for sid in source_ids:
        source_fi_dict[sid] = np.mean(
            metadata[metadata["source_id"] == sid]["trace_frequency_index"]
        )
    source_frequency_index = [
        source_fi_dict[x] for x in metadata["source_id"].to_numpy()
    ]
    metadata["source_frequency_index"] = source_frequency_index
    metadata.to_csv(metadata_path, index=False)


def extract_stead_noise(
    dest_dir: Union[str, Path],
    # split_prob=[0.75, 0.1, 0.15],
    use_all_noise=False,
    n_traces=100000,
    chunk="_STEAD_noise",
):
    dataset = sbd.STEAD(
        sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="full"
    )

    dataset.filter(
        (dataset["trace_category"] == "noise")
        & (pd.isna(dataset["trace_p_arrival_sample"]))
        & (pd.isna(dataset["trace_s_arrival_sample"])),
        inplace=True,
    )
    if (not use_all_noise) & (n_traces < len(dataset)):
        rand_idxs = np.sort(
            np.random.default_rng(seed=100).choice(
                len(dataset), size=n_traces, replace=False
            )
        )
        mask = np.zeros(len(dataset), dtype=bool)
        # mask[:n_traces] = True
        mask[rand_idxs] = True
        dataset.filter(mask, inplace=True)

    dest_dir = Path(dest_dir)
    metadata_path = dest_dir / f"metadata{chunk}.csv"
    waveforms_path = dest_dir / f"waveforms{chunk}.hdf5"
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        writer.data_format = {  # Define data format
            "dimension_order": "CW",
            "component_order": "ZNE",
            # "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        for i in range(len(dataset)):
            waveform, metadata = dataset.get_sample(i)
            event_params = {
                "source_id": metadata["source_id"],
                "source_origin_time": metadata["source_origin_time"],
                "source_latitude_deg": metadata["source_latitude_deg"],
                "source_longitude_deg": metadata["source_longitude_deg"],
                "source_depth_km": metadata["source_depth_km"],
                "source_magnitude": metadata["source_magnitude"],
                "source_magnitude_type": metadata["source_magnitude_type"],
                "source_type": "noise",
            }
            trace_params = {
                "station_network_code": metadata["station_network_code"],
                "station_code": metadata["station_code"],
                "station_location_code": None,
                "trace_channel": metadata["trace_channel"],
                "station_latitude_deg": metadata["station_latitude_deg"],
                "station_longitude_deg": metadata["station_longitude_deg"],
                "station_elevation_m": metadata["station_elevation_m"],
                "station_epicentral_distance_m": np.nan,
                "path_azimuth_deg": np.nan,
                "path_back_azimuth_deg": np.nan,
                "trace_p_arrival_time": np.nan,
                "trace_s_arrival_time": np.nan,
                "trace_p_status": metadata["trace_p_status"],
                "trace_s_status": metadata["trace_s_status"],
                "trace_p_max_weight": np.nan,
                "trace_s_max_weight": np.nan,
                "trace_p_first_motion": np.nan,
                "trace_name": metadata["trace_name_original"],
                "trace_start_time": metadata["trace_start_time"],
                "trace_has_spikes": False,
                "trace_p_arrival_sample": np.nan,
                "trace_s_arrival_sample": np.nan,
                "trace_sampling_rate_hz": metadata["trace_sampling_rate_hz"],
                "trace_frequency_index": np.nan,
                "source_frequency_index": np.nan,
                "split": metadata["split"],
                "trace_snr_db": np.nan,
                "trace_mean_snr_db": np.nan,
            }
            # # split
            # trace_params["split"] = np.random.choice(
            #     ["train", "dev", "test"], p=split_prob
            # )

            writer.add_trace({**event_params, **trace_params}, waveform)
