from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import numpy as np
import seisbench.generate as sbg
import logging
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from tqdm import tqdm

import json


def eval_tasks123(
    model,
    dataset,
    targets,
    exp_name,
    sets,
    batchsize,
    num_workers,
    output_remark: str = "pred",
    sampling_rate=None,
    root_save_dir=None,
    # exp_details={},
):
    # weights = Path(weights)
    targets = Path(targets)
    if isinstance(sets, str):
        sets = sets.split(",")

    if root_save_dir is None:
        pred_path = targets.parent / f"{targets.name}_{output_remark}" / "task123"
    else:
        root_save_dir = Path(root_save_dir)
        pred_path = root_save_dir / f"{targets.name}_{output_remark}" / "task123"
    save_dir = pred_path / exp_name
    save_dir.mkdir(exist_ok=True, parents=True)
    # if save_dir is None:
    #     save_dir = (
    #         targets.parent / f"{targets.name}_{output_remark}" / "task123" / exp_name
    #     )
    # else:
    #     save_dir = Path(save_dir)

    # # version = sorted(weights.iterdir())[-1]
    # version = sorted(list(weights.iterdir()), key=lambda x: int(x.name.split("_")[1]))[
    #     -1
    # ]
    # config_path = version / "hparams.yaml"
    # with open(config_path, "r") as f:
    #     # config = yaml.safe_load(f)
    #     config = yaml.full_load(f)

    # model_cls = models.__getattribute__(config["model"] + "Lit")
    # model = load_best_model(model_cls, weights, version.name)

    # data_name = data_aliases[targets.name]

    # if data_name != config["data"]:
    #     logging.warning("Detected cross-domain evaluation")
    #     pred_root = "pred_cross"
    #     parts = weights.name.split()
    #     weight_path_name = "_".join(parts[:2] + [targets.name] + parts[2:])
    # else:
    #     pred_root = "pred"
    #     weight_path_name = weights.name

    # dataset = data.get_dataset_by_name(data_name)(
    #     sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="full"
    # )

    # if sampling_rate is not None:
    #     dataset.sampling_rate = sampling_rate
    #     pred_root = pred_root + "_resampled"
    #     weight_path_name = weight_path_name + f"_{sampling_rate}"

    for eval_set in sets:
        split = dataset.get_split(eval_set)
        if targets.name == "instance":
            logging.warning(
                "Overwriting noise trace_names to allow correct identification"
            )
            # Replace trace names for noise entries
            split._metadata["trace_name"].values[
                -len(split.datasets[-1]) :
            ] = split._metadata["trace_name"][-len(split.datasets[-1]) :].apply(
                lambda x: "noise_" + x
            )
            split._build_trace_name_to_idx_dict()

        logging.warning(f"Starting set {eval_set}")
        split.preload_waveforms(pbar=True)

        for task in ["1", "23"]:
            task_csv = targets / f"task{task}.csv"

            if not task_csv.is_file():
                continue

            logging.warning(f"Starting task {task}")

            task_targets = pd.read_csv(task_csv)
            task_targets = task_targets[task_targets["trace_split"] == eval_set]
            if task == "1" and targets.name == "instance":
                border = _identify_instance_dataset_border(task_targets)
                task_targets["trace_name"].values[border:] = task_targets["trace_name"][
                    border:
                ].apply(lambda x: "noise_" + x)

            if sampling_rate is not None:
                for key in ["start_sample", "end_sample", "phase_onset"]:
                    if key not in task_targets.columns:
                        continue
                    task_targets[key] = (
                        task_targets[key]
                        * sampling_rate
                        / task_targets["sampling_rate"]
                    )
                task_targets[sampling_rate] = sampling_rate

            # restrict_to_phase = config.get("restrict_to_phase", None)
            # if restrict_to_phase is not None and "phase_label" in task_targets.columns:
            #     mask = task_targets["phase_label"].isin(list(restrict_to_phase))
            #     task_targets = task_targets[mask]

            # if restrict_to_phase is not None and task == "1":
            #     logging.warning("Skipping task 1 as restrict_to_phase is set.")
            #     continue

            generator = sbg.SteeredGenerator(split, task_targets)
            generator.add_augmentations(model.get_eval_augmentations())

            loader = DataLoader(
                generator, batch_size=batchsize, shuffle=False, num_workers=num_workers
            )
            trainer = pl.Trainer(accelerator="gpu", devices=1)

            predictions = trainer.predict(model, loader)

            # Merge batches
            merged_predictions = []
            for i, _ in enumerate(predictions[0]):
                merged_predictions.append(torch.cat([x[i] for x in predictions]))

            merged_predictions = [x.cpu().numpy() for x in merged_predictions]
            task_targets["score_detection"] = merged_predictions[0]
            task_targets["score_p_or_s"] = merged_predictions[1]
            task_targets["p_sample_pred"] = (
                merged_predictions[2] + task_targets["start_sample"]
            )
            task_targets["s_sample_pred"] = (
                merged_predictions[3] + task_targets["start_sample"]
            )
            save_path = save_dir / f"{eval_set}_task{task}.csv"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            task_targets.to_csv(save_path, index=False)
            # with open(save_dir / "exp_details.json", "w") as json_file:
            #     json.dump(exp_details, json_file, indent=2)


def collect_task123_results(
    path, output_fname="task123_metrics.csv", parse_filename=True
):
    """
    Traverses the given path and extracts results for each experiment

    :param path: Root path
    :param output: Path to write results csv to
    :param cross: If true, expects cross-domain results.
    :return: None
    """
    path = Path(path)

    results = []

    exp_dirs = [x for x in path.iterdir() if x.is_dir()]
    exp_dirs.sort()
    for exp_dir in tqdm(exp_dirs):
        results.append(process_exp(exp_dir, parse_filename))

    results = pd.DataFrame(results)
    if parse_filename:
        sort_keys = ["model", "lr"]
        results.sort_values(sort_keys, inplace=True)
    results.to_csv(path / output_fname, index=False)


def process_exp(exp_dir: Path, parse_filename: bool):
    """
    Extracts statistics for the given version of the given experiment.

    :param exp_dir: Path to the specific version
    :param cross: If true, expects cross-domain results.
    :return: Results dictionary
    """
    if parse_filename:
        stats = parse_exp_name(exp_dir)
    else:
        stats = {"experiment": exp_dir.name}

    stats.update(parse_task1(exp_dir))
    stats.update(parse_task23(exp_dir))

    return stats


def parse_exp_name(exp_dir):
    exp_name = exp_dir.name

    parts = exp_name.split("_")
    parts = [x for x in parts if "frac" not in x and "test" not in x]
    model_names = {"e": "eqtransformer", "p": "phasenet"}
    model = model_names[parts[0]]
    lr = float(parts[2])
    stats = {
        "experiment": exp_name,
        "model": model,
        "lr": lr,
    }

    return stats


def parse_task1(exp_dir: Path):
    if not (
        (exp_dir / "dev_task1.csv").is_file() and (exp_dir / "test_task1.csv").is_file()
    ):
        logging.warning(f"Directory {exp_dir} does not contain task 1")
        return {}

    stats = {}

    dev_pred = pd.read_csv(exp_dir / "dev_task1.csv")
    dev_pred["trace_type_bin"] = dev_pred["trace_type"] == "earthquake"
    test_pred = pd.read_csv(exp_dir / "test_task1.csv")
    test_pred["trace_type_bin"] = test_pred["trace_type"] == "earthquake"

    prec, recall, thr = precision_recall_curve(
        dev_pred["trace_type_bin"], dev_pred["score_detection"]
    )

    f1 = 2 * prec * recall / (prec + recall)
    auc = roc_auc_score(dev_pred["trace_type_bin"], dev_pred["score_detection"])

    opt_index = np.nanargmax(f1)  # F1 optimal threshold index
    opt_thr = thr[opt_index]  # F1 optimal threshold value

    dev_stats = {
        "dev_det_precision": prec[opt_index],
        "dev_det_recall": recall[opt_index],
        "dev_det_f1": f1[opt_index],
        "dev_det_auc": auc,
        "det_threshold": opt_thr,
    }
    stats.update(dev_stats)

    prec, recall, f1, _ = precision_recall_fscore_support(
        test_pred["trace_type_bin"],
        test_pred["score_detection"] > opt_thr,
        average="binary",
    )
    auc = roc_auc_score(test_pred["trace_type_bin"], test_pred["score_detection"])
    test_stats = {
        "test_det_precision": prec,
        "test_det_recall": recall,
        "test_det_f1": f1,
        "test_det_auc": auc,
    }
    stats.update(test_stats)

    return stats


def parse_task23(exp_dir: Path):
    if not (
        (exp_dir / "dev_task23.csv").is_file()
        and (exp_dir / "test_task23.csv").is_file()
    ):
        logging.warning(f"Directory {exp_dir} does not contain tasks 2 and 3")
        return {}

    stats = {}

    dev_pred = pd.read_csv(exp_dir / "dev_task23.csv")
    dev_pred["phase_label_bin"] = dev_pred["phase_label"] == "P"
    test_pred = pd.read_csv(exp_dir / "test_task23.csv")
    test_pred["phase_label_bin"] = test_pred["phase_label"] == "P"

    def add_aux_columns(pred):
        for col in ["s_sample_pred", "score_p_or_s"]:
            if col not in pred.columns:
                pred[col] = np.nan

    add_aux_columns(dev_pred)
    add_aux_columns(test_pred)

    def nanmask(pred):
        """
        Returns all entries that are nan in score_p_or_s, p_sample_pred and s_sample_pred
        """
        mask = np.logical_and(
            np.isnan(pred["p_sample_pred"]), np.isnan(pred["s_sample_pred"])
        )
        mask = np.logical_and(mask, np.isnan(pred["score_p_or_s"]))
        return mask

    if nanmask(dev_pred).all():
        logging.warning(f"{exp_dir} contains NaN predictions for tasks 2 and 3")
        return {}

    dev_pred = dev_pred[~nanmask(dev_pred)]
    test_pred = test_pred[~nanmask(test_pred)]

    skip_task2 = False
    if (
        np.logical_or(
            np.isnan(dev_pred["score_p_or_s"]), np.isinf(dev_pred["score_p_or_s"])
        ).all()
        or np.logical_or(
            np.isnan(test_pred["score_p_or_s"]), np.isinf(test_pred["score_p_or_s"])
        ).all()
    ):
        # For unfortunate combinations of nans and infs, otherwise weird scores can occur
        skip_task2 = True

    # Clipping removes infinitely likely P waves, usually resulting from models trained without S arrivals
    dev_pred["score_p_or_s"] = np.clip(dev_pred["score_p_or_s"].values, -1e100, 1e100)
    test_pred["score_p_or_s"] = np.clip(test_pred["score_p_or_s"].values, -1e100, 1e100)

    dev_pred_restricted = dev_pred[~np.isnan(dev_pred["score_p_or_s"])]
    test_pred_restricted = test_pred[~np.isnan(test_pred["score_p_or_s"])]
    if len(dev_pred_restricted) > 0 and not skip_task2:
        prec, recall, thr = precision_recall_curve(
            dev_pred_restricted["phase_label_bin"], dev_pred_restricted["score_p_or_s"]
        )

        f1 = 2 * prec * recall / (prec + recall)

        opt_index = np.nanargmax(f1)  # F1 optimal threshold index
        opt_thr = thr[opt_index]  # F1 optimal threshold value

        # Determine (approximately) optimal MCC threshold using 50 candidates
        mcc_thrs = np.sort(dev_pred["score_p_or_s"].values)
        mcc_thrs = mcc_thrs[np.linspace(0, len(mcc_thrs) - 1, 50, dtype=int)]
        mccs = []
        for thr in mcc_thrs:
            mccs.append(
                matthews_corrcoef(
                    dev_pred["phase_label_bin"], dev_pred["score_p_or_s"] > thr
                )
            )
        mcc = np.max(mccs)
        mcc_thr = mcc_thrs[np.argmax(mccs)]

        dev_stats = {
            "dev_phase_precision": prec[opt_index],
            "dev_phase_recall": recall[opt_index],
            "dev_phase_f1": f1[opt_index],
            "phase_threshold": opt_thr,
            "dev_phase_mcc": mcc,
            "phase_threshold_mcc": mcc_thr,
        }
        stats.update(dev_stats)

        prec, recall, f1, _ = precision_recall_fscore_support(
            test_pred_restricted["phase_label_bin"],
            test_pred_restricted["score_p_or_s"] > opt_thr,
            average="binary",
        )
        mcc = matthews_corrcoef(
            test_pred["phase_label_bin"], test_pred["score_p_or_s"] > mcc_thr
        )
        test_stats = {
            "test_phase_precision": prec,
            "test_phase_recall": recall,
            "test_phase_f1": f1,
            "test_phase_mcc": mcc,
        }
        stats.update(test_stats)

    for pred, set_str in [(dev_pred, "dev"), (test_pred, "test")]:
        for i, phase in enumerate(["P", "S"]):
            pred_phase = pred[pred["phase_label"] == phase]
            pred_col = f"{phase.lower()}_sample_pred"

            if len(pred_phase) == 0:
                continue

            diff = (pred_phase[pred_col] - pred_phase["phase_onset"]) / pred_phase[
                "sampling_rate"
            ]

            stats[f"{set_str}_{phase}_mean_s"] = np.mean(diff)
            stats[f"{set_str}_{phase}_median_s"] = np.median(diff)
            stats[f"{set_str}_{phase}_rmse_s"] = np.sqrt(np.mean(diff**2))
            stats[f"{set_str}_{phase}_mae_s"] = np.mean(np.abs(diff))

            bound = 1
            modified_diff = diff[(diff < bound) & (diff > -bound)]
            out_fraction = len(diff[(diff > bound) | (diff < -bound)]) / len(diff)
            modified_mae = np.mean(np.abs(modified_diff))
            modified_rmse = np.sqrt(np.mean(modified_diff**2))

            stats[f"{set_str}_{phase}_out_s"] = out_fraction
            stats[f"{set_str}_{phase}_modified_rmse_s"] = modified_rmse
            stats[f"{set_str}_{phase}_modified_mae_s"] = modified_mae

    return stats


def _identify_instance_dataset_border(task_targets):
    """
    Calculates the dataset border between Signal and Noise for instance,
    assuming it is the only place where the bucket number does not increase
    """
    buckets = task_targets["trace_name"].apply(lambda x: int(x.split("$")[0][6:]))

    last_bucket = 0
    for i, bucket in enumerate(buckets):
        if bucket < last_bucket:
            return i
        last_bucket = bucket
