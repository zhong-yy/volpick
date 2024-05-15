from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pathlib import Path
import os
import time
import datetime
import yaml
from ast import literal_eval
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

import logging

from tqdm import tqdm

import volpick.model.models as models
import seisbench.models as sbm
import seisbench.generate as sbg
from volpick.model.utils import load_best_model_from_base_dir, load_best_model
import seisbench.data as sbd
from volpick.model.generate_eval_targets import (
    generate_task0,
    generate_task1,
    generate_task23,
)

from volpick.model.models import phase_dict
from volpick.model.eval_taks0 import (
    evaluate,
    get_ground_truth,
    count_TP_FP_FN,
    compute_residuals,
    calculate_precision_recall_F1score,
    eval_task0,
    opt_prob_metrics,
    collect_task0_results,
    collect_task0_results_dev,
)
from volpick.model.eval_taks123 import eval_tasks123, collect_task123_results

import json

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from obspy import read, Stream, Trace
from obspy.signal.trigger import ar_pick

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events


class ARpicker:
    def __init__(
        self, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s, windowlen=3001
    ):
        self.set_params(f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)
        self.windowlen = windowlen

    def set_params(self, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s):
        self.f1 = f1
        self.f2 = max(f1 + 2, f2)  # ensure f2>f1

        self.sta_p = sta_p
        self.lta_p = max(sta_p + 0.5, lta_p)

        self.sta_s = sta_s
        self.lta_s = max(sta_s + 0.5, lta_s)

        self.m_p = m_p
        self.m_s = m_s
        self.l_p = l_p
        self.l_s = l_s

    def get_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=self.windowlen, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict(self, sample):
        p_relative_to_p0 = -1
        s_relative_to_p0 = -1
        p0, p1 = sample[f"window_borders"]
        p_pick, s_pick = ar_pick(
            a=sample["X"][0],
            b=sample["X"][1],
            c=sample["X"][2],
            samp_rate=100,
            f1=self.f1,
            f2=self.f2,
            lta_p=self.lta_p,
            sta_p=self.sta_p,
            lta_s=self.lta_s,
            sta_s=self.sta_s,
            m_p=self.m_p,
            m_s=self.m_s,
            l_p=self.l_p,
            l_s=self.l_s,
        )
        # print(p0)
        p_relative_to_p0 = int(p_pick * 100) - p0
        s_relative_to_p0 = int(s_pick * 100) - p0
        return p_relative_to_p0, s_relative_to_p0

    @classmethod
    def load_from_log(cls, path):
        opt = -np.inf
        params = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    pass
                parsed_line = json.loads(line)
                if parsed_line["target"] > opt:
                    opt = parsed_line["target"]
                    params = parsed_line["params"]

        logging.warning(f"Optimal value: {- opt:.1f}\tParams: {params}")
        return cls(**params)


limit = 2500
np.random.seed(42)

targets_path = Path("/home/zhongyiyuan/volpick/model_training/Eval_targets/all")

# steered metatdata
task_csv = targets_path / "task0.csv"
targets = pd.read_csv(task_csv)
targets = targets[targets["trace_split"] == "dev"]
targets = targets[targets["source_type"] != "noise"]

trace_names = None
if len(targets) > limit:
    trace_names = targets["trace_name"].values.copy()
    np.random.shuffle(trace_names)
    trace_names = trace_names[:limit]
    mask = targets["trace_name"].isin(set(trace_names))
    targets = targets[mask]

data_path = Path("/home/zhongyiyuan/DATA/my_datasets_seisbench/vcseis")
dataset = sbd.WaveformDataset(
    data_path,
    sampling_rate=100,
    component_order="ZNE",
    dimension_order="NCW",
    cache="full",
)
dataset.filter(dataset["split"].isin(["dev"]), inplace=True)
if trace_names is not None:
    mask = dataset.metadata["trace_name"].isin(set(trace_names))
    dataset.filter(mask, inplace=True)
dataset.preload_waveforms(pbar=True)

picker = ARpicker(
    f1=1.0,
    f2=25,
    lta_p=2.0,
    sta_p=0.2,
    lta_s=8.0,
    sta_s=2.0,
    m_p=2,
    m_s=8,
    l_p=0.25,
    l_s=0.5,
)

generator = sbg.SteeredGenerator(dataset, targets)
generator.add_augmentations(picker.get_augmentations())

# p_relative_to_p0, s_relative_to_p0 = picker.predict(generator[185])

pbar = False
picker = ARpicker(
    f1=0,
    f2=0,
    lta_p=0,
    sta_p=0,
    lta_s=0,
    sta_s=0,
    m_p=0,
    m_s=0,
    l_p=0,
    l_s=0,
)


def fitness(f1, f2, lta_p, sta_p, lta_s, sta_s, l_p, l_s):
    picker.set_params(f1, f2, lta_p, sta_p, lta_s, sta_s, 2, 8, l_p, l_s)

    generator = sbg.SteeredGenerator(dataset, targets)
    generator.add_augmentations(picker.get_augmentations())

    p_preds = []
    s_preds = []
    if pbar:
        itr = tqdm(range(len(generator)), total=len(generator))
    else:
        itr = range(len(generator))

    for i in itr:
        print(i)
        p_relative_to_p0, s_relative_to_p0 = picker.predict(generator[i])
        print(p_relative_to_p0, s_relative_to_p0)
        row = targets.iloc[i]

        p_pred = row["start_sample"] + p_relative_to_p0
        s_pred = row["start_sample"] + s_relative_to_p0
        print(p_pred, s_pred)
        p_preds.append(p_pred)
        s_preds.append(s_pred)

    p_preds = np.array(p_preds)
    s_preds = np.array(s_preds)

    rmse = np.sqrt(
        np.nanmean((p_preds - targets["trace_p_arrival_sample"].values) ** 2)
    ) + np.sqrt(
        np.nanmean((s_preds - targets["trace_s_arrival_sample"].values) ** 2)
    )  # RMSE in samples
    print(-rmse)

    return -rmse  # Negative as optimizer is built for maximizing the target function


fitness(
    **{
        "f1": 1,
        "f2": 25,
        "lta_p": 2,
        "sta_p": 0.1,
        "lta_s": 8.0,
        "sta_s": 1.0,
        "l_p": 0.25,
        "l_s": 0.5,
    }
)
