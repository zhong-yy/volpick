import numpy as np
from scipy.signal import windows
from scipy.fft import fft, fftfreq
import pandas as pd
from obspy import UTCDateTime, read
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.fft import fft, fftfreq
from scipy import signal

from pathlib import Path
import shutil
from obspy.imaging.cm import viridis_white

import seisbench.data as sbd

# from obspy.imaging.spectrogram import spectrogram
from matplotlib import mlab
import seisbench.models as sbm
import logging
import volpick
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


def freqency_index(data, dt, low_freq_band, high_freq_band):
    """
    Get the frequency index from a spectrum
    """
    # Fast fourier transform
    N = len(data)
    win = windows.hann(N)
    spec = fft(data * win)[0 : N // 2]
    freq = fftfreq(N, dt)[0 : N // 2]

    # Calculate frequency index
    ind_upper = np.logical_and(freq > high_freq_band[0], freq < high_freq_band[1])
    Aupper = np.mean(np.abs(spec)[ind_upper])
    ind_lower = np.logical_and(freq > low_freq_band[0], freq < low_freq_band[1])
    Alower = np.mean(np.abs(spec)[ind_lower])
    return np.log10(Aupper / Alower)


def calculate_snr(data, p_sample, s_sample, winlen):
    n_samples = data.shape[-1]
    # print(f"p sample: {p_sample}, s sample: {s_sample}")
    if (p_sample is None) or (p_sample < 10):
        return [np.nan, np.nan, np.nan], np.nan
    elif (s_sample is not None) and (s_sample < n_samples - 10):
        snrs = []
        for i in range(3):
            # print(int(p_sample - winlen), int(p_sample))
            noi = np.percentile(
                np.abs(
                    data[
                        i,
                        max(0, int(p_sample - winlen)) : int(p_sample),
                    ]
                ),
                95,
            )
            sig = np.percentile(
                np.abs(
                    data[
                        i,
                        int(s_sample) : min(int(s_sample + winlen), n_samples),
                    ]
                ),
                95,
            )
            if np.isclose(noi, 0) or np.isclose(sig, 0):
                snrs.append(np.nan)
            else:
                snrs.append(20 * np.log10(sig / noi))
        return snrs, np.nanmean(snrs)
    else:
        snrs = []
        for i in range(3):
            noi = np.percentile(
                np.abs(
                    data[
                        i,
                        max(0, int(p_sample - winlen)) : int(p_sample),
                    ]
                ),
                95,
            )
            sig = np.percentile(
                np.abs(
                    data[
                        i,
                        int(p_sample) : min(int(p_sample + winlen), n_samples),
                    ]
                ),
                95,
            )
            if np.isclose(noi, 0) or np.isclose(sig, 0):
                snrs.append(np.nan)
            else:
                snrs.append(20 * np.log10(sig / noi))
        return snrs, np.nanmean(snrs)


def extract_events(trace_metadata: pd.DataFrame):
    column_labels = trace_metadata.columns
    drop_labels = [x for x in column_labels if "source" not in x]

    events = trace_metadata.drop_duplicates(
        subset="source_id", keep="first", ignore_index=True
    ).copy()
    events.sort_values(by="source_origin_time", ignore_index=True, inplace=True)
    events.drop(labels=drop_labels, axis=1, inplace=True)
    return events


def generate_chunk_file(data_dir):
    data_dir = Path(data_dir)
    metadata_files = list(data_dir.glob("*.csv"))
    chunks = []
    for file in metadata_files:
        chunks.append(file.name[8:-4])
    chunks.sort()
    with open(data_dir / "chunks", "w") as f:
        for chunk in chunks:
            f.write(chunk + "\n")


def assemble_datasets(datasets_dir, datasets, dest_dir):
    dest_dir = Path(dest_dir)
    try:
        dest_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{dest_dir} exists.")
    for data_folder in datasets:
        data_folder = Path(datasets_dir) / data_folder
        print(data_folder)
        for data_file in data_folder.iterdir():
            shutil.copy2(data_file, dest_dir)


# def exclude_close_events(
#     whole_catalog: pd.DataFrame,
#     target_catalog: pd.DataFrame,
#     org_diff_threshold: float = 70,
#     lat_diff_threshold: float = 0.5,
#     lon_diff_threshold: float = 0.5,
# ):
#     all_events = extract_events(whole_catalog)
#     all_org = all_events["source_origin_time"].apply(lambda x: UTCDateTime(x))
#     # forward difference
#     all_forward_diff_org = np.append(np.diff(all_org.values), np.nan)
#     # backward difference
#     all_backward_diff_org = np.insert(np.diff(all_org.values), 0, np.nan)

#     assert (all_org.iloc[2] - all_org.iloc[1]) == all_forward_diff_org[1]
#     assert (all_org.iloc[1] - all_org.iloc[0]) == all_backward_diff_org[1]
#     all_events["forward_diff_org"] = all_forward_diff_org
#     all_events["backward_diff_org"] = all_backward_diff_org

#     all_forward_diff_lat = np.append(
#         np.diff(all_events["source_latitude_deg"].values), np.nan
#     )
#     all_backward_diff_lat = np.insert(
#         np.diff(all_events["source_latitude_deg"].values), 0, np.nan
#     )

#     all_forward_diff_lon = np.append(
#         np.diff(all_events["source_longitude_deg"].values), np.nan
#     )
#     all_backward_diff_lon = np.insert(
#         np.diff(all_events["source_longitude_deg"].values), 0, np.nan
#     )

#     all_events["forward_diff_lat"] = all_forward_diff_lat
#     all_events["backward_diff_lat"] = all_backward_diff_lat

#     all_events["forward_diff_lon"] = all_forward_diff_lon
#     all_events["backward_diff_lon"] = all_backward_diff_lon

#     close_events = all_events[
#         (
#             (all_events["forward_diff_org"] < org_diff_threshold)
#             & (all_events["forward_diff_lat"].abs() < lat_diff_threshold)
#             & (all_events["forward_diff_lon"].abs() < lon_diff_threshold)
#         )
#         | (
#             (all_events["backward_diff_org"] < org_diff_threshold)
#             & (all_events["backward_diff_lat"].abs() < lat_diff_threshold)
#             & (all_events["backward_diff_lon"].abs() < lon_diff_threshold)
#         )
#     ]

#     result = target_catalog[
#         ~(target_catalog["source_id"].isin(close_events["source_id"].tolist()))
#     ].copy()

#     result.reset_index(inplace=True, drop=True)

#     return result


def plot_waveforms(
    waveform_table,
    data_dir,
    indices,
    fig_dir=None,
    dpi=300,
    savefig=True,
    xrangemin=None,
    xrangemax=None,
    fts=10,
    **kwargs,
):
    if max(indices) > len(waveform_table):
        raise KeyError(
            f"The maximum requested index {max(indices)} is larger than" f"the number"
        )
    data_dir = Path(data_dir)
    if fig_dir is None:
        fig_dir = data_dir.parent / f"{data_dir.name}_fig"
    print(f"Plotting {len(indices)} figures")
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
        phase_colors = {"P": "blue", "S": "red"}
        for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
            if pd.notna(metadata[phase]):
                picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

        plt.subplots_adjust(hspace=0.1)
        for k in range(nc):
            x_times = st[k].times(reftime=starttime)
            y_data = st[k].data
            axs[k][0].plot(
                st[k].times(reftime=starttime),
                st[k].data,
                label=st[k].id,
                color="black",
                linewidth=1,
                **kwargs,
            )
            if xrangemin is None:
                xmin = min(st[k].times(reftime=starttime)[0] for k in range(nc))
            else:
                xmin = min(picks)[0] + xrangemin - starttime
            if xrangemax is None:
                xmax = max(st[k].times(reftime=starttime)[-1] for k in range(nc))
            else:
                xmax = max(picks)[0] + xrangemax - starttime
            axs[k][0].set_xlim(xmin, xmax)

            ymin = np.min(y_data[(x_times >= xmin) & (x_times <= xmax)])
            ymax = np.max(y_data[(x_times >= xmin) & (x_times <= xmax)])
            ylength = ymax - ymin
            ymin = ymin - ylength * 0.15
            ymax = ymax + ylength * 0.15
            axs[k][0].set_ylim(ymin, ymax)

            for pick, phase_label in picks:
                axs[k][0].vlines(
                    pick - starttime,
                    ymin=ymin,
                    ymax=ymax,
                    color=phase_colors[phase_label],
                    label=phase_label,
                )
            axs[k][0].legend()
            axs[k][0].tick_params(labelsize=fts)
        axs[-1][0].set_xlabel(f"Time from {starttime}", fontsize=fts)
        if savefig:
            plt.savefig(
                fig_dir / metadata["trace_name"].replace("mseed", "jpg"),
                bbox_inches="tight",
                dpi=dpi,
            )
        plt.close(fig)


def plot_spectrum(
    waveform_table,
    data_dir,
    indices,
    fig_dir=None,
    dpi=300,
    savefig=True,
    xrangemin=None,
    xrangemax=None,
    fts=10,
    **kwargs,
):
    if max(indices) > len(waveform_table):
        raise KeyError(
            f"The maximum requested index {max(indices)} is larger than" f"the number"
        )
    data_dir = Path(data_dir)
    if fig_dir is None:
        fig_dir = data_dir.parent / f"{data_dir.name}_fig"
    print(f"Plotting {len(indices)} figures")
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True, exist_ok=False)
    for i in indices:
        metadata = waveform_table.iloc[i]
        st = read(data_dir / metadata["trace_name"])
        st.detrend("demean").detrend("linear")
        st.merge(method=1, fill_value=0)
        starttime = min(trace.stats.starttime for trace in st)
        st.merge()
        nc = len(st)
        cm = 1 / 2.54
        fig, axs = plt.subplots(
            nc,
            2,
            figsize=(19 * cm, nc * 3.5 * cm),
            sharex="col",
            squeeze=False,
        )
        picks = []
        phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
        phase_colors = {"P": "blue", "S": "red"}
        for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
            if pd.notna(metadata[phase]):
                picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

        plt.subplots_adjust(hspace=0.1, wspace=0.05)
        st_cp = st.copy()
        for k, tr in enumerate(st_cp):
            axs[k][0].plot(
                tr.times(reftime=starttime),
                tr.data,
                # label=tr.id,
                color="black",
                linewidth=0.5,
                **kwargs,
            )
            ymin, ymax = axs[k][0].get_ylim()

            if len(picks) > 0:
                for pick, phase_label in picks:
                    axs[k][0].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                    )
                axs[k][0].legend(fontsize=fts)
            if xrangemin is None:
                xmin = min(tr.times(reftime=starttime)[0] for tr in st_cp)
            else:
                xmin = min(picks)[0] + xrangemin - starttime
            if xrangemax is None:
                xmax = max(tr.times(reftime=starttime)[-1] for tr in st_cp)
            else:
                xmax = max(picks)[0] + xrangemax - starttime
            axs[k][0].set_xlim(xmin, xmax)
            axs[k][0].text(
                0.97,
                0.02,
                tr.id,
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[k][0].transAxes,
                fontsize=9,
            )

            # spectrum
            window = signal.hann(tr.stats.npts)
            yf = fft(tr.data * window)
            freqs = fftfreq(tr.stats.npts, tr.stats.delta)[: tr.stats.npts // 2]
            axs[k][1].axvline(1, color="gray")
            axs[k][1].semilogx(
                freqs[freqs > 0.1],
                np.abs(yf[: tr.stats.npts // 2][freqs > 0.1]),
                color="blue",
                linewidth=0.5,
            )

            axs[k][1].set_xlim([min(freqs[freqs > 0.1]), max(freqs[freqs > 0.1])])
            axs[k][1].yaxis.tick_right()
            # axs[k][1].ticklabel_format(
            #     axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            # )
            axs[k][1].yaxis.get_offset_text().set_fontsize(7)
            axs[k][0].tick_params(labelsize=fts)
            axs[k][1].tick_params(labelsize=fts)

        axs[0][0].set_title("Time series", fontsize=fts)
        axs[-1][0].set_xlabel(f"Time from {starttime}", fontsize=fts)
        axs[0][1].set_title("Amplitude spectrum", fontsize=fts)
        axs[-1][1].set_xlabel("Frequency (Hz)", fontsize=fts)
        if savefig:
            plt.savefig(
                fig_dir / metadata["trace_name"].replace("mseed", "jpg"),
                bbox_inches="tight",
                dpi=dpi,
            )
        plt.close(fig)


def plot_spectrogram(
    waveform_table,
    data_dir,
    indices,
    fig_dir=None,
    dpi=300,
    savefig=True,
    xrangemin=None,
    xrangemax=None,
    fts=10,
    **kwargs,
):
    if max(indices) > len(waveform_table):
        raise KeyError(
            f"The maximum requested index {max(indices)} is larger than" f"the number"
        )
    data_dir = Path(data_dir)
    if fig_dir is None:
        fig_dir = data_dir.parent / f"{data_dir.name}_fig"
    print(f"Plotting {len(indices)} figures")
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True, exist_ok=False)
    for i in indices:
        metadata = waveform_table.iloc[i]
        st = read(data_dir / metadata["trace_name"])
        starttime = min(trace.stats.starttime for trace in st)
        st.detrend("demean").detrend("linear")
        st.merge(method=1, fill_value=0)
        nc = len(st)
        cm = 1 / 2.54
        fig, axs = plt.subplots(
            nc,
            3,
            figsize=(14, nc * 3.0),
            sharex="col",
            squeeze=False,
        )
        picks = []
        phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
        phase_colors = {"P": "blue", "S": "red"}
        for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
            if pd.notna(metadata[phase]):
                picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        st_cp = st.copy()
        # st_cp.detrend("demean").detrend("linear")
        st_cp.filter("highpass", freq=0.3)
        for k, tr in enumerate(st_cp):
            axs[k][0].plot(
                tr.times(reftime=starttime),
                tr.data,
                # label=tr.id,
                color="black",
                linewidth=0.5,
                **kwargs,
            )
            ymin, ymax = axs[k][0].get_ylim()

            if len(picks) > 0:
                for pick, phase_label in picks:
                    axs[k][0].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                        linewidth=1,
                    )
                axs[k][0].legend(fontsize=fts)
            if xrangemin is None:
                xmin = min(tr.times(reftime=starttime)[0] for tr in st_cp)
            else:
                xmin = min(picks)[0] + xrangemin - starttime
            if xrangemax is None:
                xmax = max(tr.times(reftime=starttime)[-1] for tr in st_cp)
            else:
                xmax = max(picks)[0] + xrangemax - starttime
            axs[k][0].set_xlim(xmin, xmax)
            axs[k][0].text(
                0.97,
                0.02,
                tr.id,
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[k][0].transAxes,
                fontsize=9,
            )
            axs[k][0].ticklabel_format(
                axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            )

            # spectrum
            window = signal.hann(tr.stats.npts)
            yf = fft(tr.data * window)
            freqs = fftfreq(tr.stats.npts, tr.stats.delta)[: tr.stats.npts // 2]
            axs[k][1].axvline(1, color="gray")
            axs[k][1].semilogx(
                freqs[freqs > 0.1],
                np.abs(yf[: tr.stats.npts // 2][freqs > 0.1]),
                color="blue",
                linewidth=0.5,
            )

            axs[k][1].set_xlim([min(freqs[freqs > 0.3]), max(freqs[freqs > 0.3])])
            # axs[k][1].yaxis.tick_right()
            # axs[k][1].ticklabel_format(
            #     axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            # )
            axs[k][1].yaxis.get_offset_text().set_fontsize(7)

            # n_per_segment = 128
            # n_overlap = int(0.9 * n_per_segment)

            # f, t, Sxx = signal.spectrogram(
            #     tr.data,
            #     fs=tr.stats.sampling_rate,
            #     mode="magnitude",
            #     nperseg=n_per_segment,
            #     noverlap=n_overlap,
            # )
            # pm = axs[k][2].pcolormesh(t, f, Sxx, shading="gouraud", cmap="jet")
            # plt.colorbar(pm, ax=axs[k][2])
            _, ax_img = spectrogram(
                data=tr.data,
                samp_rate=tr.stats.sampling_rate,
                axes=axs[k][2],
                cmap="jet"
                # dbscale=True,
            )
            clb = fig.colorbar(
                ax_img,
                ax=axs[k][2],
                location="bottom",
                pad=0.22,
                shrink=0.9,
                aspect=25,
            )
            clb.ax.tick_params(labelsize=fts)

            axs[k][2].set_yscale("log")
            axs[k][2].set_ylim([0.3, 50])
            # axs[k][2].yaxis.tick_right()
            axs[k][2].set_ylabel("Frequency (Hz)", fontsize=fts, color="white")
            axs[k][2].yaxis.set_label_coords(0.1, 0.5)
            axs[k][2].set_xlabel("Time (s)", fontsize=fts)

            axs[k][0].tick_params(labelsize=fts)
            axs[k][1].tick_params(labelsize=fts)

        axs[0][0].set_title("Time series", fontsize=fts)
        axs[-1][0].set_xlabel(f"Time from {starttime}", fontsize=fts)
        axs[0][1].set_title("Amplitude spectrum", fontsize=fts)
        axs[-1][1].set_xlabel("Frequency (Hz)", fontsize=fts)
        if savefig:
            plt.savefig(
                fig_dir / metadata["trace_name"].replace("mseed", "jpg"),
                bbox_inches="tight",
                dpi=dpi,
            )
        plt.close(fig)


def check_waveforms_parallel(
    waveform_table,
    data_dir,
    fig_dir=None,
    dpi=300,
    xrangemin=None,
    xrangemax=None,
    num_processes=16,
    phasenet=sbm.PhaseNet.from_pretrained("original"),
    eqtransformer=sbm.EQTransformer.from_pretrained("original"),
    skip_threshold=1,
    fts=9,
    **kwargs,
):
    data_dir = Path(data_dir)
    if fig_dir is None:
        fig_dir = data_dir.parent / f"{data_dir.name}_fig"
    print(f"Plotting {len(waveform_table)} figures")
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True, exist_ok=False)

    ctx = mp.get_context("spawn")
    mp_start_method = ctx.get_start_method()
    catalog_chunks = []
    chunksize = len(waveform_table) // num_processes
    print(f"Chunk size: {chunksize}")
    assert chunksize >= 2, (
        f"{num_processes} processes are used. Start method: {mp_start_method}. Chunk size is {chunksize}."
        f"Please try using less process"
    )
    for i in range(num_processes - 1):
        catalog_chunks.append(waveform_table.iloc[:chunksize].copy())
        waveform_table.drop(waveform_table.index[:chunksize], inplace=True)

    catalog_chunks.append(waveform_table.copy())
    waveform_table.drop(waveform_table.index[:], inplace=True)
    process_list = []
    proc_names = []
    for i in range(num_processes):
        proc_name = f"_p{i}"
        proc_names.append(proc_name)
        proc = ctx.Process(
            target=check_waveform,
            kwargs={
                "waveform_table": catalog_chunks[i],
                "data_dir": data_dir,
                "fig_dir": fig_dir,
                "dpi": dpi,
                "savefig": True,
                "xrangemin": xrangemin,
                "xrangemax": xrangemax,
                "phasenet": phasenet,
                "eqtransformer": eqtransformer,
                "skip_threshold": skip_threshold,
                "fts": fts,
                **kwargs,
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


def check_waveform(
    waveform_table,
    data_dir,
    fig_dir=None,
    dpi=300,
    savefig=True,
    xrangemin=None,
    xrangemax=None,
    phasenet=sbm.PhaseNet.from_pretrained("original"),
    eqtransformer=sbm.EQTransformer.from_pretrained("original"),
    fts=9,
    skip_threshold=1,
    **kwargs,
):
    if mp.parent_process() is None:
        process_mark = ""
        print("It is the main process")
    else:
        process_mark = mp.current_process().name

    # Initialize a logger
    logger = volpick.logger.getChild("plot" + process_mark)
    logger.setLevel(logging.INFO)

    data_dir = Path(data_dir)
    if fig_dir is None:
        fig_dir = data_dir.parent / f"{data_dir.name}_fig"
    if not fig_dir.exists():
        fig_dir.mkdir(parents=True, exist_ok=False)
    for i in range(len(waveform_table)):
        metadata = waveform_table.iloc[i]
        st = read(data_dir / metadata["trace_name"])
        starttime = min(trace.stats.starttime for trace in st)
        st.detrend("demean").detrend("linear")
        st.merge(method=1, fill_value=0)
        # st.taper(max_percentage=0.025, type="hann")
        nc = len(st)
        cm = 1 / 2.54
        fig, axs = plt.subplots(
            4,
            nc + 1,
            figsize=((nc + 1) * 5.2, 9.5),
            sharex="col",
            squeeze=False,
            width_ratios=[1.2] * nc + [1],
            height_ratios=[1, 1, 1.25, 1],
        )
        plt.subplots_adjust(hspace=0.125, wspace=0.15)
        picks = []
        phase_hints = {"trace_p_arrival_time": "P", "trace_s_arrival_time": "S"}
        phase_colors = {"P": "blue", "S": "red"}
        for phase in ["trace_p_arrival_time", "trace_s_arrival_time"]:
            if pd.notna(metadata[phase]):
                picks.append((UTCDateTime(metadata[phase]), phase_hints[phase]))

        st_cp = st.copy()

        st_cp_filtered = st.copy()
        # st_cp.detrend("demean").detrend("linear")

        st_cp.filter("highpass", freq=0.3)

        st_cp_filtered.filter("bandpass", freqmin=1, freqmax=20)
        # assert st_cp is not st_cp_filtered

        try:
            annotations1 = eqtransformer.annotate(st_cp)
            # offset = annotations1[0].stats.starttime - starttime
            skip = False
            for id_compon in range(3):
                if np.max(annotations1[id_compon].data) > skip_threshold:
                    skip = True
                    break
                axs[0][-1].plot(
                    annotations1[id_compon].times(reftime=starttime),
                    annotations1[id_compon].data,
                    label=annotations1[id_compon]
                    .stats.channel.replace("_", " ")
                    .replace("EQTransformer", "EQT"),
                )
            if skip:
                logger.warning(f"""{i}: Skip {metadata["trace_name"]}""")
                plt.close(fig)
                continue
            # axs[0][-1].set_ylim([-0.05, 1.095])
            # axs[0][-1].legend(fontsize=fts)

            annotations2 = phasenet.annotate(st_cp)
            # offset = annotations2[0].stats.starttime - starttime
            for id_compon in range(3):
                if annotations2[id_compon].stats.channel != "PhaseNet_N":
                    if np.max(annotations2[id_compon].data) > skip_threshold:
                        skip = True
                        break
                    axs[0][-1].plot(
                        annotations2[id_compon].times(reftime=starttime),
                        annotations2[id_compon].data,
                        label=annotations2[id_compon].stats.channel.replace("_", " "),
                        linestyle="--",
                    )
            if skip:
                logger.warning(f"""{i}: Skip {metadata["trace_name"]}""")
                plt.close(fig)
                continue
            axs[0][-1].set_ylim([-0.025, 1.0999])
            axs[0][-1].legend(
                fontsize=fts,
                ncols=2,
                frameon=False,
                columnspacing=1,
                labelspacing=0.3,
                loc="upper right",
            )
            # axs[1][-1].set_xlabel(f"Time from {starttime} [s]", fontsize=fts)
            axs[0][-1].xaxis.set_minor_locator(MultipleLocator(5))
            # axs[1][-1].xaxis.set_minor_locator(MultipleLocator(5))
            axs[0][-1].tick_params(labelsize=fts)
            # axs[1][-1].tick_params(labelsize=fts)
            axs[0][-1].text(
                0.01,
                0.95,
                ">0.3Hz",
                verticalalignment="top",
                horizontalalignment="left",
                transform=axs[0][-1].transAxes,
                fontsize=fts,
                color="black",
            )
        except Exception:
            logger.error(
                f"""{i}: Error in using phasenet or eqtransformer. Continue. {metadata["trace_name"]}"""
            )
            plt.close(fig)
            continue
        try:
            annotations1 = eqtransformer.annotate(st_cp_filtered)
            # offset = annotations1[0].stats.starttime - starttime
            for id_compon in range(3):
                if np.max(annotations1[id_compon].data) > skip_threshold:
                    skip = True
                    break
                axs[1][-1].plot(
                    annotations1[id_compon].times(reftime=starttime),
                    annotations1[id_compon].data,
                    label=annotations1[id_compon]
                    .stats.channel.replace("_", " ")
                    .replace("EQTransformer", "EQT"),
                )
            if skip:
                logger.warning(f"""{i}: Skip {metadata["trace_name"]}""")
                plt.close(fig)
                continue
            annotations2 = phasenet.annotate(st_cp_filtered)
            # offset = annotations2[0].stats.starttime - starttime
            for id_compon in range(3):
                if annotations2[id_compon].stats.channel != "PhaseNet_N":
                    if np.max(annotations2[id_compon].data) > skip_threshold:
                        skip = True
                        break
                    axs[1][-1].plot(
                        annotations2[id_compon].times(reftime=starttime),
                        annotations2[id_compon].data,
                        label=annotations2[id_compon].stats.channel.replace("_", " "),
                        linestyle="--",
                    )
            if skip:
                logger.warning(f"""{i}: Skip {metadata["trace_name"]}""")
                plt.close(fig)
                continue
            axs[1][-1].set_ylim([-0.025, 1.0999])
            axs[1][-1].legend(
                fontsize=fts,
                ncols=2,
                frameon=False,
                columnspacing=1,
                labelspacing=0.3,
                loc="upper right",
            )
            axs[1][-1].xaxis.set_minor_locator(MultipleLocator(5))
            axs[1][-1].tick_params(labelsize=fts)
            axs[1][-1].text(
                0.01,
                0.95,
                "1-20Hz",
                verticalalignment="top",
                horizontalalignment="left",
                transform=axs[1][-1].transAxes,
                fontsize=fts,
                color="black",
            )
        except Exception:
            logger.error(
                f"""{i}: Error in using phasenet or eqtransformer. Continue. {metadata["trace_name"]}"""
            )
            plt.close(fig)
            continue

        for k, tr in enumerate(st_cp):
            axs[0][k].plot(
                tr.times(reftime=starttime),
                tr.data,
                # label=tr.id,
                color="black",
                linewidth=0.5,
                **kwargs,
            )
            axs[1][k].plot(
                st_cp_filtered[k].times(reftime=starttime),
                st_cp_filtered[k].data,
                # label=tr.id,
                color="black",
                linewidth=0.5,
                **kwargs,
            )
            # assert not np.allclose(st_cp_filtered[k], tr)

            if xrangemin is None:
                xmin = min(tr.times(reftime=starttime)[0] for tr in st_cp)
            else:
                xmin = min(picks)[0] + xrangemin - starttime
            if xrangemax is None:
                xmax = max(tr.times(reftime=starttime)[-1] for tr in st_cp)
            else:
                xmax = max(picks)[0] + xrangemax - starttime
            # ymin, ymax = axs[0][k].get_ylim()
            x_times = tr.times(reftime=starttime)
            ymin = np.min(tr.data[(x_times >= xmin) & (x_times <= xmax)])
            ymax = np.max(tr.data[(x_times >= xmin) & (x_times <= xmax)])
            ylength = ymax - ymin

            ymin2 = np.min(
                st_cp_filtered[k].data[(x_times >= xmin) & (x_times <= xmax)]
            )
            ymax2 = np.max(
                st_cp_filtered[k].data[(x_times >= xmin) & (x_times <= xmax)]
            )
            ylength2 = ymax2 - ymin2
            ymin2 = ymin2 - ylength2 * 0.125
            ymax2 = ymax2 + ylength2 * 0.125

            ymin = ymin - ylength * 0.125
            ymax = ymax + ylength * 0.125
            if ylength > 1e-5:
                axs[0][k].set_ylim(ymin, ymax)
            if ylength2 > 1e-5:
                axs[1][k].set_ylim(ymin2, ymax2)

            axs[0][k].set_xlim(xmin, xmax)
            axs[1][k].set_xlim(xmin, xmax)

            if len(picks) > 0:
                for pick, phase_label in picks:
                    axs[0][k].vlines(
                        pick - starttime,
                        ymin=ymin,
                        ymax=ymax,
                        color=phase_colors[phase_label],
                        label=phase_label,
                        linewidth=1,
                    )
                    axs[1][k].vlines(
                        pick - starttime,
                        ymin=ymin2,
                        ymax=ymax2,
                        color=phase_colors[phase_label],
                        label=phase_label,
                        linewidth=1,
                    )
                axs[0][k].legend(fontsize=fts)
                axs[1][k].legend(fontsize=fts)

            axs[0][k].text(
                0.97,
                0.02,
                tr.id + " (>0.3Hz)",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[0][k].transAxes,
                fontsize=fts,
            )
            axs[0][k].ticklabel_format(
                axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            )
            axs[1][k].text(
                0.97,
                0.02,
                tr.id + " (1-20Hz)",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[1][k].transAxes,
                fontsize=fts,
            )
            axs[0][k].ticklabel_format(
                axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            )
            axs[1][k].ticklabel_format(
                axis="y", style="scientific", scilimits=(-2, 2), useMathText=True
            )
            axs[0][k].yaxis.get_offset_text().set_fontsize(fts)
            axs[1][k].yaxis.get_offset_text().set_fontsize(fts)

            # try:
            #     cft = recursive_sta_lta(
            #         tr.data,
            #         min(int(5 * tr.stats.sampling_rate), len(tr.data)),
            #         min(int(20.0 * tr.stats.sampling_rate), len(tr.data)),
            #     )
            #     on_of = trigger_onset(cft, 2.0, 0.5)
            # except Exception:
            #     logger.error("Error in calculating STA/LTA. Continue")
            #     continue
            # axs[1][k].plot(tr.times(reftime=starttime), cft, label="Recursive STA/LTA")
            # axs[1][k].hlines([2.0, 0.5], 0, len(cft), color=["r", "b"], linestyle="--")
            # ymin, ymax = axs[1][k].get_ylim()
            # if len(on_of) > 0:
            #     axs[1][k].vlines(
            #         on_of[:, 0] / tr.stats.sampling_rate
            #         + tr.times(reftime=starttime)[0],
            #         ymin,
            #         ymax,
            #         color="r",
            #         linewidth=2,
            #     )
            #     axs[1][k].vlines(
            #         on_of[:, 1] / tr.stats.sampling_rate
            #         + tr.times(reftime=starttime)[0],
            #         ymin,
            #         ymax,
            #         color="b",
            #         linewidth=2,
            #     )

            _, ax_img = spectrogram(
                data=tr.data,
                samp_rate=tr.stats.sampling_rate,
                wlen=256 / tr.stats.sampling_rate,
                axes=axs[2][k],
                # cmap=viridis_white,
                # dbscale=True,
            )
            clb = fig.colorbar(
                ax_img,
                ax=axs[2][k],
                location="top",
                pad=0.03,
                shrink=0.8,
                aspect=50,
            )
            axs[2][k].set_yscale("log")
            axs[2][k].set_ylim([0.3, 50])
            clb.ax.tick_params(labelsize=fts)
            clb.ax.ticklabel_format(
                style="scientific",
                scilimits=(-5, 5),
                useMathText=True,
            )
            clb.ax.xaxis.get_offset_text().set_fontsize(fts)
            axs[2][k].text(
                0.97,
                0.02,
                ">0.3Hz",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[2][k].transAxes,
                fontsize=fts,
                color="white",
            )

            # 1-20Hz
            _, ax_img = spectrogram(
                data=st_cp_filtered[k].data,
                samp_rate=st_cp_filtered[k].stats.sampling_rate,
                wlen=256 / st_cp_filtered[k].stats.sampling_rate,
                axes=axs[3][k],
                # cmap=viridis_white,
                # dbscale=True,
            )
            clb = fig.colorbar(
                ax_img,
                ax=axs[3][k],
                location="top",
                pad=0.03,
                shrink=0.8,
                aspect=50,
            )
            axs[3][k].set_yscale("log")
            axs[3][k].set_ylim([1, 20])
            clb.ax.tick_params(labelsize=fts)
            clb.ax.ticklabel_format(
                style="scientific",
                scilimits=(-5, 5),
                useMathText=True,
            )
            clb.ax.xaxis.get_offset_text().set_fontsize(fts)
            axs[3][k].text(
                0.97,
                0.02,
                "1-20Hz",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=axs[3][k].transAxes,
                fontsize=fts,
                color="white",
            )
            axs[-1][k].set_xlabel(f"Time from {starttime} [s]", fontsize=fts)

            axs[0][k].tick_params(labelsize=fts)
            axs[1][k].tick_params(labelsize=fts)
            axs[2][k].tick_params(labelsize=fts)
            axs[3][k].tick_params(labelsize=fts)

            axs[0][k].xaxis.set_minor_locator(MultipleLocator(5))
            axs[1][k].xaxis.set_minor_locator(MultipleLocator(5))
            axs[2][k].xaxis.set_minor_locator(MultipleLocator(5))
            axs[3][k].xaxis.set_minor_locator(MultipleLocator(5))
        # axs[1][0].set_ylabel("STA/LTA", fontsize=fts)
        axs[2][0].set_ylabel("Frequency (Hz)", fontsize=fts)
        axs[3][0].set_ylabel("Frequency (Hz)", fontsize=fts)
        # fig.delaxes(axs[-1][-1])
        # fig.delaxes(axs[-2][-1])
        # axs[1][0].yaxis.set_label_coords(0.1, 0.5)

        for tr in st_cp:
            try:
                cft = recursive_sta_lta(
                    tr.data,
                    min(int(5 * tr.stats.sampling_rate), len(tr.data)),
                    min(int(20.0 * tr.stats.sampling_rate), len(tr.data)),
                )
                on_of = trigger_onset(cft, 2, 0.5)
            except Exception:
                logger.error(f"{i}: Error in calculating STA/LTA. Continue")
                break
            axs[2][-1].plot(tr.times(reftime=starttime), cft, label=tr.stats.channel)
            # axs[2][-1].hlines(
            #     [2.0, 0.5], 0, len(cft), color=["r", "b"], linestyle="--"
            # )
            axs[2][-1].axhline(2, color="red", linestyle="--")
            axs[2][-1].axhline(0.5, color="blue", linestyle="--")
            ymin, ymax = axs[2][-1].get_ylim()
            if len(on_of) > 0:
                axs[2][-1].vlines(
                    on_of[:, 0] / tr.stats.sampling_rate
                    + tr.times(reftime=starttime)[0],
                    0,
                    2.5,
                    color="r",
                    linewidth=1,
                )
                axs[2][-1].vlines(
                    on_of[:, 1] / tr.stats.sampling_rate
                    + tr.times(reftime=starttime)[0],
                    0,
                    2.5,
                    color="b",
                    linewidth=1,
                )
        axs[2][-1].legend(fontsize=fts, ncols=3, loc="lower right")
        axs[2][-1].set_ylabel("Recursive STA/LTA", fontsize=fts)
        axs[2][-1].xaxis.set_minor_locator(MultipleLocator(5))
        axs[2][-1].tick_params(labelsize=fts)
        axs[2][-1].text(
            0.01,
            0.97,
            ">0.3Hz",
            verticalalignment="top",
            horizontalalignment="left",
            transform=axs[2][-1].transAxes,
            fontsize=fts,
            color="black",
        )

        for tr in st_cp_filtered:
            try:
                cft = recursive_sta_lta(
                    tr.data,
                    min(int(5 * tr.stats.sampling_rate), len(tr.data)),
                    min(int(20.0 * tr.stats.sampling_rate), len(tr.data)),
                )
                on_of = trigger_onset(cft, 2, 0.5)
            except Exception:
                logger.error("Error in calculating STA/LTA. Continue")
                break
            axs[3][-1].plot(tr.times(reftime=starttime), cft, label=tr.stats.channel)
            axs[3][-1].axhline(2, color="red", linestyle="--")
            axs[3][-1].axhline(0.5, color="blue", linestyle="--")
            ymin, ymax = axs[3][-1].get_ylim()
            if len(on_of) > 0:
                axs[3][-1].vlines(
                    on_of[:, 0] / tr.stats.sampling_rate
                    + tr.times(reftime=starttime)[0],
                    0,
                    2.5,
                    color="r",
                    linewidth=1,
                )
                axs[3][-1].vlines(
                    on_of[:, 1] / tr.stats.sampling_rate
                    + tr.times(reftime=starttime)[0],
                    0,
                    2.5,
                    color="b",
                    linewidth=1,
                )
        axs[3][-1].legend(fontsize=fts, ncols=3, loc="lower right")
        axs[3][-1].set_ylabel("Recursive STA/LTA", fontsize=fts)
        axs[3][-1].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[3][-1].tick_params(labelsize=fts)
        axs[3][-1].set_xlim(xmin, xmax)
        axs[3][-1].text(
            0.01,
            0.97,
            "1-20Hz",
            verticalalignment="top",
            horizontalalignment="left",
            transform=axs[3][-1].transAxes,
            fontsize=fts,
            color="black",
        )
        # axs[3][-1].set_xlabel(f"Time from {starttime} [s]", fontsize=fts)
        # axs[3][-1].xaxis.set_minor_locator(MultipleLocator(5))
        # axs[3][-1].tick_params(labelsize=fts)

        if savefig:
            figure_name = metadata["trace_name"].replace("mseed", "jpg")
            plt.savefig(
                fig_dir / figure_name,
                bbox_inches="tight",
                dpi=dpi,
            )
            logger.info(f"{i}: Saved {figure_name}")
        plt.close(fig)


def get_dataset_by_name(name):
    """
    Resolve dataset name to class from seisbench.data.

    :param name: Name of dataset as defined in seisbench.data.
    :return: Dataset class from seisbench.data
    """
    try:
        return sbd.__getattribute__(name)
    except AttributeError:
        raise ValueError(f"Unknown dataset '{name}'.")


def get_dataset_by_path(data_path):
    return sbd.WaveformDataset(
        Path(data_path),
        sampling_rate=100,
        component_order="ZNE",
        dimension_order="NCW",
        cache="full",
    )


# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
#  Modification: return an axeimage for plotting a colorbar
#  Modified by: Yiyuan Zhong, 2023 Sep. 05
#
#  Purpose: Plotting spectrogram of Seismograms.
#   Author: Christian Sippl, Moritz Beyreuther
#    Email: sippl@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Christian Sippl
# --------------------------------------------------------------------
"""
Plotting spectrogram of seismograms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import os
import math

import numpy as np
from matplotlib import mlab
from matplotlib.colors import Normalize

from obspy.imaging.cm import obspy_sequential


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrogram(
    data,
    samp_rate,
    per_lap=0.9,
    wlen=None,
    log=False,
    outfile=None,
    fmt=None,
    axes=None,
    dbscale=False,
    mult=8.0,
    cmap=obspy_sequential,
    zorder=None,
    title=None,
    show=True,
    clip=[0.0, 1.0],
):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to a
        window length matching 128 samples.
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """
    import matplotlib.pyplot as plt

    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (
            f"Input signal too short ({npts} samples, window length "
            f"{wlen} seconds, nfft {nfft} samples, sampling rate "
            f"{samp_rate} Hz)"
        )
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(
        data, Fs=samp_rate, NFFT=nfft, pad_to=mult, noverlap=nlap
    )

    if len(time) < 2:
        msg = (
            f"Input signal too short ({npts} samples, window length "
            f"{wlen} seconds, nfft {nfft} samples, {nlap} samples window "
            f"overlap, sampling rate {samp_rate} Hz)"
        )
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (("cmap", cmap), ("zorder", zorder)) if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale("log")
        # Plot times
        ax_img = ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (
            time[0] - halfbin_time,
            time[-1] + halfbin_time,
            freq[0] - halfbin_freq,
            freq[-1] + halfbin_freq,
        )
        ax_img = ax.imshow(specgram, interpolation="nearest", extent=extent, **kwargs)

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis("tight")
    ax.set_xlim(0, end)
    ax.grid(False)

    if axes:
        return ax, ax_img

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    if title:
        ax.set_title(title)

    if not os.environ.get("SPHINXBUILD"):
        # ignoring all NumPy warnings during plot
        with np.errstate(all="ignore"):
            plt.draw()
    if outfile:
        if fmt:
            fig.savefig(outfile, format=fmt)
        else:
            fig.savefig(outfile)
    elif show:
        plt.show()
    else:
        return fig
