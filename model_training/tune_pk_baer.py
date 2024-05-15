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
from obspy.signal.trigger import ar_pick, pk_baer

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events


class BaerKradolfer:
    def __init__(self, lp, hp, tdownmax, tupevent, thr1, windowlen=3000):
        self.lp = lp
        self.hp = hp
        self.tdownmax = tdownmax
        self.tupevent = tupevent
        self.thr1 = thr1
        self.windowlen = windowlen

    def get_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=self.windowlen, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    def predict(self, sample):
        pred_relative_to_p0 = -1

        x = sample[f"X"][0]  # Remove channel axis
        p0, p1 = sample[f"window_borders"]
        p_pick, _ = pk_baer(
            x,
            100,
            int(100 * self.tdownmax),
            int(100 * self.tupevent),
            self.thr1,
            2 * self.thr1,
            100,
            100,
        )
        pred_relative_to_p0 = p_pick - p0

        return pred_relative_to_p0

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


def tune_baer(targets_path, pbar, limit=2500):
    np.random.seed(42)
    targets_path = Path("/home/zhongyiyuan/volpick/model_training/Eval_targets/all")

    # steered metatdata
    task_csv = targets_path / "task0.csv"
    targets = pd.read_csv(task_csv)
    targets = targets[targets["trace_split"] == "dev"]
    targets = targets[pd.notna(targets["trace_p_arrival_sample"])]
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

    # fitness
    bounds = {
        "lp": (1, 49),
        "hp": (0.001, 5),
        "tdownmax": (0.5, 15),
        "tupevent": (0.3, 3),
        "thr1": (1, 40),
    }

    picker = BaerKradolfer(0, 0, 0, 0, 0)  # Parameters will anyhow be overwritten

    def fitness(lp, hp, tdownmax, tupevent, thr1):
        """
        Fitness function for Bayesian optimization

        :param lp: Lowpass frequency
        :param hp: Highpass frequency
        :param tdownmax: See obspy.signal.trigger.pk_bear
        :param tupevent: See obspy.signal.trigger.pk_bear
        :param thr1: See obspy.signal.trigger.pk_bear . We also set thr2 = 2 * thr1
        :return:
        """
        picker.lp = lp
        picker.hp = hp
        picker.tdownmax = tdownmax
        picker.tupevent = tupevent
        picker.thr1 = thr1

        generator = sbg.SteeredGenerator(dataset, targets)
        generator.add_augmentations(picker.get_augmentations())

        preds = []
        if pbar:
            itr = tqdm(range(len(generator)), total=len(generator))
        else:
            itr = range(len(generator))

        for i in itr:
            pred_relative_to_p0 = picker.predict(generator[i])
            row = targets.iloc[i]

            pred = row["start_sample"] + pred_relative_to_p0
            preds.append(pred)

        preds = np.array(preds)
        rmse = np.sqrt(
            np.mean((preds - targets["trace_p_arrival_sample"].values) ** 2)
        )  # RMSE in samples

        return (
            -rmse
        )  # Negative as optimizer is built for maximizing the target function

    # optimize
    optimizer = BayesianOptimization(
        f=fitness,
        pbounds=bounds,
        random_state=1,
    )
    # Setup loggers
    os.makedirs("baer_logs", exist_ok=True)
    logger = JSONLogger(path=f"baer_logs/{targets_path.name}.json")
    screen_logger = ScreenLogger()
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, screen_logger)

    # Roughly matching entries from NMSOP
    optimizer.probe(
        params={"lp": 49, "hp": 3, "tdownmax": 2, "tupevent": 1.5, "thr1": 10},
        lazy=True,
    )
    optimizer.probe(
        params={"lp": 5, "hp": 1, "tdownmax": 3, "tupevent": 1.5, "thr1": 5}, lazy=True
    )
    optimizer.probe(
        params={"lp": 2, "hp": 0.1, "tdownmax": 5, "tupevent": 1.5, "thr1": 2},
        lazy=True,
    )

    optimizer.maximize(init_points=25, n_iter=500)

    print(optimizer.max)


def eval_baer(targets_path, root_save_dir):
    np.random.seed(42)
    # targets_path = Path("/home/zhongyiyuan/volpick/model_training/Eval_targets/all")

    # steered metatdata
    task_csv = targets_path / "task0.csv"
    targets = pd.read_csv(task_csv)
    # targets = targets[targets["trace_split"] == "test"]
    targets = targets[pd.notna(targets["trace_p_arrival_sample"])]
    targets = targets[targets["source_type"] != "noise"]

    model_path = Path("baer_logs/all.json")
    model = BaerKradolfer.load_from_log(model_path)

    data_path = Path("/home/zhongyiyuan/DATA/my_datasets_seisbench/vcseis")
    dataset = sbd.WaveformDataset(
        data_path,
        sampling_rate=100,
        component_order="ZNE",
        dimension_order="NCW",
        cache="full",
    )

    split = dataset.get_split("test")
    split.preload_waveforms(pbar=True)

    split_targets = targets[targets["trace_split"] == "test"].copy()

    generator = sbg.SteeredGenerator(split, split_targets)
    generator.add_augmentations(model.get_augmentations())
    preds = []
    itr = tqdm(range(len(generator)), total=len(generator))
    for i in itr:
        pred_relative_to_p0 = model.predict(generator[i])
        preds.append(pred_relative_to_p0)

    split_targets["p_sample_pred"] = preds + split_targets["start_sample"]

    TP_p, FP_p, FN_p, tps_p, fps_p, fns_p = count_TP_FP_FN(
        p_idxs_ground_truth,
        p_idxs_predicted,
        tp_thre=tp_thre,
        sampling_rate=100,
        method=count_tp_method,
    )
    precision_p, recall_p, F1score_p = calculate_precision_recall_F1score(
        TP_p, FP_p, FN_p
    )
    
    root_save_dir = Path(root_save_dir)
    pred_path = (
        root_save_dir
        / f"{targets_path.name}_pred"
        / "task0"
        / "baer"
        / "test_task0.csv"
    )
    # pred_path = Path("pred_baer") / f"{source}_baer_{target}" / f"{eval_set}_task23.csv"
    pred_path.parent.mkdir(exist_ok=True, parents=True)
    split_targets.to_csv(pred_path, index=False)


if __name__ == "__main__":
    targets_path = Path("/home/zhongyiyuan/volpick/model_training/Eval_targets/all")
    tune_baer(targets_path=targets_path, pbar=True, limit=2500)
