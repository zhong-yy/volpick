import volpick.data
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import time
from pathlib import Path
import datetime
import logging
import volpick
from volpick.data import NoiseData, AlaskaDataset, HawaiiDataset
import numpy as np
from volpick.data.utils import (
    plot_waveforms,
    plot_spectrum,
    plot_spectrogram,
    check_waveform,
    check_waveforms_parallel,
)


def plot_LP_ak(method=1):
    alsk = volpick.data.AlaskaDataset()
    catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")
    lp_catalog_table = catalog_table[catalog_table["source_type"] == "lp"]

    data_dir = alsk.save_dir / "mseed"
    # catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")
    # data_dir = alsk.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(lp_catalog_table), size=500, replace=False
    )
    if method == 1:
        plot_spectrogram(
            lp_catalog_table,
            data_dir,
            random_idxs,
            fig_dir=alsk.save_dir / f"{len(random_idxs)}lp_figures",
        )
    elif method == 2:
        check_waveform(
            lp_catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=alsk.save_dir / f"{len(random_idxs)}lp_figures2",
        )
    # plot_spectrogram(
    #     lp_catalog_table,
    #     data_dir,
    #     random_idxs,
    #     fig_dir=alsk.save_dir / f"{len(random_idxs)}lp_figures",
    # )


def plot_LP_hw(method=1):
    hawaii = volpick.data.HawaiiDataset(
        root_folder_name="hawaii2012to2021",
        cat_file_name="hawaii_catalog2012to2021",
    )
    catalog_table = pd.read_csv(hawaii.save_dir / "mseed_log" / "downloads.csv")
    lp_catalog_table = catalog_table[catalog_table["source_type"] == "lp"]

    data_dir = hawaii.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(lp_catalog_table), size=500, replace=False
    )

    if method == 1:
        plot_spectrogram(
            lp_catalog_table,
            data_dir,
            random_idxs,
            fig_dir=hawaii.save_dir / f"{len(random_idxs)}lp_figures",
        )
    elif method == 2:
        check_waveform(
            lp_catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=hawaii.save_dir / f"{len(random_idxs)}lp_figures2",
        )


def plot_VT_ak(method=1):
    alsk = volpick.data.AlaskaDataset()
    catalog_table = pd.read_csv(alsk.save_dir / "mseed_log" / "downloads.csv")
    vt_catalog_table = catalog_table[catalog_table["source_type"] != "lp"]

    data_dir = alsk.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(vt_catalog_table), size=500, replace=False
    )

    if method == 1:
        plot_spectrogram(
            vt_catalog_table,
            data_dir,
            random_idxs,
            fig_dir=alsk.save_dir / f"{len(random_idxs)}vt_figures",
        )
    elif method == 2:
        check_waveform(
            vt_catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=alsk.save_dir / f"{len(random_idxs)}vt_figures2",
        )


def plot_VT_hw(method=1):
    hawaii = volpick.data.HawaiiDataset(
        root_folder_name="hawaii2012to2021",
        cat_file_name="hawaii_catalog2012to2021",
    )
    catalog_table = pd.read_csv(hawaii.save_dir / "mseed_log" / "downloads.csv")
    vt_catalog_table = catalog_table[catalog_table["source_type"] != "lp"]

    data_dir = hawaii.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(vt_catalog_table), size=500, replace=False
    )

    if method == 1:
        plot_spectrogram(
            vt_catalog_table,
            data_dir,
            random_idxs,
            fig_dir=hawaii.save_dir / f"{len(random_idxs)}vt_figures",
        )
    elif method == 2:
        check_waveform(
            vt_catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=hawaii.save_dir / f"{len(random_idxs)}vt_figures2",
        )


def plot_VT_jp(method=1):
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "VT")
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    data_dir = japan.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(catalog_table), size=500, replace=False
    )
    if method == 1:
        plot_spectrogram(
            catalog_table,
            data_dir,
            random_idxs,
            fig_dir=japan.save_dir / f"{len(random_idxs)}vt_figures",
        )
    elif method == 2:
        check_waveform(
            catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=japan.save_dir / f"{len(random_idxs)}vt_figures2",
        )


def plot_LP_jp(method=1):
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "Vol_lp")
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    data_dir = japan.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(catalog_table), size=500, replace=False
    )
    if method == 1:
        plot_spectrogram(
            catalog_table,
            data_dir,
            random_idxs,
            fig_dir=japan.save_dir / f"{len(random_idxs)}lp_figures",
        )
    elif method == 2:
        check_waveform(
            catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=japan.save_dir / f"{len(random_idxs)}lp_figures2",
        )


def plot_tecLP_jp(method=1):
    japan = volpick.data.JapanDataset(save_dir=volpick.cache_root / "Japan" / "Tec_lp")
    catalog_table = pd.read_csv(japan.save_dir / "mseed_log" / "downloads.csv")
    data_dir = japan.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(catalog_table), size=100, replace=False
    )
    if method == 1:
        plot_spectrogram(
            catalog_table,
            data_dir,
            random_idxs,
            fig_dir=japan.save_dir / f"{len(random_idxs)}lp_figures",
        )
    elif method == 2:
        check_waveform(
            catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=japan.save_dir / f"{len(random_idxs)}lp_figures2",
        )


def plot_lp_pnsn_lp(
    method=1,
):
    pnsn = volpick.data.ComCatDataset(
        root_folder_name="Cascade_lp", cat_file_name="cascade_catalog_lp"
    )
    catalog_table = pd.read_csv(pnsn.save_dir / "mseed_log" / "downloads.csv")
    data_dir = pnsn.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(catalog_table), size=100, replace=False
    )
    if method == 1:
        plot_spectrogram(
            catalog_table,
            data_dir,
            random_idxs,
            fig_dir=pnsn.save_dir / f"{len(random_idxs)}lp_figures",
        )
    elif method == 2:
        check_waveform(
            catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=pnsn.save_dir / f"{len(random_idxs)}lp_figures2",
        )


def plot_lp_pnsn_vt(
    method=1,
):
    pnsn = volpick.data.ComCatDataset(
        root_folder_name="Cascade_vt", cat_file_name="cascade_catalog_vt"
    )
    catalog_table = pd.read_csv(pnsn.save_dir / "mseed_log" / "downloads.csv")
    data_dir = pnsn.save_dir / "mseed"
    random_idxs = np.random.default_rng(seed=51).choice(
        len(catalog_table), size=100, replace=False
    )
    if method == 1:
        plot_spectrogram(
            catalog_table,
            data_dir,
            random_idxs,
            fig_dir=pnsn.save_dir / f"{len(random_idxs)}vt_figures",
        )
    elif method == 2:
        check_waveform(
            catalog_table.iloc[random_idxs],
            data_dir,
            fig_dir=pnsn.save_dir / f"{len(random_idxs)}vt_figures2",
        )


if __name__ == "__main__":
    # plot_LP_jp(2)
    # plot_VT_jp(2)
    # plot_LP_hw(2)
    # plot_VT_hw(2)
    # plot_LP_ak(2)
    # plot_VT_ak(2)
    # plot_tecLP_jp(2)
    plot_lp_pnsn_lp(
        1,
    )
    # plot_lp_pnsn_lp(
    #     2,
    # )
    plot_lp_pnsn_vt(
        1,
    )
    # plot_lp_pnsn_vt(
    #     2,
    # )
