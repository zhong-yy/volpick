from torch.utils.data import DataLoader
import torch
from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import numpy as np

import seisbench.generate as sbg
from obspy.signal.trigger import trigger_onset
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
import logging


def evaluate(
    generator: sbg.SteeredGenerator,
    model,
    threshold: float,
    batchsize: int,
    num_workers: int,
    save_prob: bool = False,
    show_pbar: bool = False,
):
    """
    Make predictions on data specified by a SteeredGenerator

    Note: for phasenet model, the model.labels must be the same as the the order of P,S,N used in
    training set
    """
    dataloader = DataLoader(
        generator, batch_size=batchsize, shuffle=False, num_workers=num_workers
    )

    if isinstance(threshold, list):
        p_threshold = threshold[0]
        s_threshold = threshold[1]
    else:
        p_threshold = threshold
        s_threshold = threshold

    def get_picks_from_prob(prob, threshold):
        triggers = trigger_onset(prob, thres1=threshold, thres2=threshold / 2.0)
        scores = []
        picks = []
        # get picks from annotations
        for s0, s1 in triggers:
            peak_value = np.max(prob[s0 : s1 + 1])
            idx_peak = s0 + np.argmax(prob[s0 : s1 + 1])
            scores.append(peak_value)
            picks.append(idx_peak)
        return np.array(scores), np.array(picks)

    model.eval()
    predictions = []
    probs = []
    with torch.no_grad():
        if show_pbar:
            pbar = tqdm(total=len(generator), desc="Predicting")  # show progress bar
        for batch_id, batch in enumerate(dataloader):
            window_borders = batch["window_borders"]
            current_batch_size = len(batch["X"])
            # predict
            if model.name == "EQTransformer":
                det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
                det_pred = det_pred.cpu().numpy()
                p_pred = p_pred.cpu().numpy()
                s_pred = s_pred.cpu().numpy()
            # elif model.name == "VolEQTransformer":
            #     rg_det_pred, lp_det_pred, p_pred, s_pred = model(
            #         batch["X"].to(model.device)
            #     )
            #     rg_det_pred = rg_det_pred.cpu().numpy()
            #     lp_det_pred = lp_det_pred.cpu().numpy()
            #     p_pred = p_pred.cpu().numpy()
            #     s_pred = s_pred.cpu().numpy()

            #     batch_score_rg_detection = [[] for _ in range(current_batch_size)]
            #     batch_score_lp_detection = [[] for _ in range(current_batch_size)]
            #     batch_score_lp_over_rg = [[] for _ in range(current_batch_size)]
            elif model.name == "PhaseNet":
                batch_preds = model(
                    batch["X"].to(model.device)
                )  # pred shape: batch size x 3 x npts
                batch_preds = batch_preds.cpu().numpy()

            batch_p_picks = [[] for _ in range(current_batch_size)]
            batch_p_scores = [[] for _ in range(current_batch_size)]
            batch_s_picks = [[] for _ in range(current_batch_size)]
            batch_s_scores = [[] for _ in range(current_batch_size)]

            for i in range(current_batch_size):
                # local_start_sample is the local index in the target window
                local_start_sample, local_end_sample = window_borders[i]
                if model.name == "EQTransformer":
                    local_pred = np.array(
                        [
                            p_pred[i, local_start_sample:local_end_sample],
                            s_pred[i, local_start_sample:local_end_sample],
                            det_pred[i, local_start_sample:local_end_sample],
                        ]
                    )
                    p_label_index = 0
                    s_label_index = 1
                # elif model.name == "VolEQTransformer":
                #     local_rg_det_pred = rg_det_pred[
                #         i, local_start_sample:local_end_sample
                #     ]
                #     local_lp_det_pred = lp_det_pred[
                #         i, local_start_sample:local_end_sample
                #     ]
                #     local_pred = np.array(
                #         [
                #             p_pred[i, local_start_sample:local_end_sample],
                #             s_pred[i, local_start_sample:local_end_sample],
                #             local_rg_det_pred,
                #             local_lp_det_pred,
                #         ]
                #     )
                #     batch_score_rg_detection[i] = np.max(local_rg_det_pred)
                #     batch_score_lp_detection[i] = np.max(local_lp_det_pred)
                #     batch_score_lp_over_rg[i] = (
                #         batch_score_lp_detection[i] / batch_score_rg_detection[i]
                #     )
                #     p_label_index = 0
                #     s_label_index = 1
                elif model.name == "PhaseNet":
                    local_pred = batch_preds[i, :, local_start_sample:local_end_sample]
                    p_label_index = model.labels.index("P")
                    s_label_index = model.labels.index("S")
                batch_p_scores[i], batch_p_picks[i] = get_picks_from_prob(
                    local_pred[p_label_index], p_threshold
                )
                batch_s_scores[i], batch_s_picks[i] = get_picks_from_prob(
                    local_pred[s_label_index], s_threshold
                )
                if save_prob:
                    probs.append(local_pred)

            # 把p_pick, p_score, s_pick, s_score放在一个元组, 万一以后需要改到pytorch-lightning中, 更容易修改
            # predictions is a list of tuples of lists of lists, (number of batches, number of metrics(4), batch size, number of picks (variable))
            # if model.name == "VolEQTransformer":
            #     predictions.append(
            #         (
            #             batch_p_picks,
            #             batch_p_scores,
            #             batch_s_picks,
            #             batch_s_scores,
            #             batch_score_rg_detection,
            #             batch_score_lp_detection,
            #             batch_score_lp_over_rg,
            #         )
            #     )
            # else:
            predictions.append(
                (batch_p_picks, batch_p_scores, batch_s_picks, batch_s_scores)
            )
            if show_pbar:
                pbar.update(batch["X"].shape[0])  # update progress bar
        if show_pbar:
            pbar.close()  # remember to close progress bar

    n_output = 4
    # if model.name == "VolEQTransformer":
    #     n_output = 7
    merged_predictions = [
        [] for _ in range(n_output)
    ]  # 0: p_pick_sample, 1: p_score, 2: s_pick_sample, 3: s_score
    for i in range(n_output):
        merged_predictions[i] = [
            item for batch_preds in predictions for item in batch_preds[i]
        ]
    for idx in range(len(merged_predictions[0])):
        # merged_predictions[0][idx] is np.ndarray, so broadcasting is used
        merged_predictions[0][idx] = (
            merged_predictions[0][idx] + generator.metadata.iloc[idx]["start_sample"]
        )
        merged_predictions[2][idx] = (
            merged_predictions[2][idx] + generator.metadata.iloc[idx]["start_sample"]
        )
    if save_prob:
        merged_predictions.append(probs)
    #     return tuple(
    #         *merged_predictions,
    #         probs,
    #     )
    #     # return (
    #     #     merged_predictions[0],
    #     #     merged_predictions[1],
    #     #     merged_predictions[2],
    #     #     merged_predictions[3],
    #     #     probs,
    #     # )
    # else:
    #     return tuple(*merged_predictions)
    return merged_predictions


def get_ground_truth(
    generator: sbg.SteeredGenerator,
    p_arrival_sample_column="trace_p_arrival_sample",
    s_arrival_sample_column="trace_s_arrival_sample",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Retrieve ground truths from the generator
    """
    p_idxs_ground_truth = [np.array([]) for _ in range(len(generator))]
    s_idxs_ground_truth = [np.array([]) for _ in range(len(generator))]
    for idx in range(len(generator)):
        # get true picks (unit: sample)
        control = generator.metadata.iloc[idx].to_dict()
        kwargs = {
            "trace_name": control["trace_name"],
            "chunk": control.get("trace_chunk", None),
            "dataset": control.get("trace_dataset", None),
        }
        data_idx = generator.dataset.get_idx_from_trace_name(**kwargs)
        try:
            p_truth = generator.dataset.get_sample(data_idx)[1][p_arrival_sample_column]
        except KeyError:
            p_truth = np.nan
        try:
            s_truth = generator.dataset.get_sample(data_idx)[1][s_arrival_sample_column]
        except KeyError:
            s_truth = np.nan

        start_sample = generator.metadata.iloc[idx]["start_sample"]
        end_sample = generator.metadata.iloc[idx]["end_sample"]
        if not np.isnan(p_truth):
            if p_truth >= start_sample and p_truth < end_sample:
                p_idxs_ground_truth[idx] = np.array([p_truth])
        if not np.isnan(s_truth):
            if s_truth >= start_sample and s_truth < end_sample:
                s_idxs_ground_truth[idx] = np.array([s_truth])
    return p_idxs_ground_truth, s_idxs_ground_truth


def count_TP_FP_FN(
    picks_truth: list[np.ndarray],
    picks_predicted: list[np.ndarray],
    tp_thre: float = 0.5,
    sampling_rate: float = 100,
    method=0,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Count the total numbers of TP, FP and FN for all traces,
     as well as the numbers of them for each individual trace
    """
    tps = np.zeros(len(picks_predicted))
    fps = np.zeros(len(picks_predicted))
    fns = np.zeros(len(picks_predicted))

    if method == 0:
        for i in range(len(tps)):
            if len(picks_predicted[i]) == 0:
                # For this trace, the number of false negative equals to the number of true picks (which is always 1 or 0 in my dataset)
                fns[i] += len(picks_truth[i])
            else:  # else: model outputs some picks
                if len(picks_truth[i]) == 0:  # actually no phases
                    fps[i] += len(picks_predicted[i])  #
                else:
                    # looking at "true" picks to count true positives and false negatives
                    for current_pick_truth in picks_truth[i]:
                        # tps[i]+=np.sum((picks_predicted[i]-pick)/sampling_rate <=tp_thre)
                        if np.any(
                            np.abs(
                                (picks_predicted[i] - current_pick_truth)
                                / sampling_rate
                            )
                            <= tp_thre
                        ):
                            tps[i] += 1  # true positive
                        else:
                            fns[i] += 1  # false negative

                    # count false positives
                    for pick_pred in picks_predicted[i]:
                        # Do not count true positives again
                        if np.all(
                            np.abs((pick_pred - picks_truth[i]) / sampling_rate)
                            > tp_thre
                        ):
                            fps[i] += 1
    elif method == 1:
        for i in range(len(tps)):
            if len(picks_predicted[i]) == 0:  # negative prediction
                if len(picks_truth[i]) > 0:  # positive ground truth
                    fns[i] += 1
            else:  # positive prediction
                if len(picks_truth[i]) == 0:  # actually no phases
                    fps[i] += 1
                else:
                    assert len(picks_truth[i]) == 1
                    # for current_pick_truth in picks_truth[i]:
                    if np.any(
                        np.abs((picks_predicted[i] - picks_truth[i][0]) / sampling_rate)
                        <= tp_thre
                    ):
                        tps[i] += 1  # true positive
                    else:  # this case is ambiguous
                        fps[i] += 1
                        # else:
                        #     fns[i] += 1  # false negative
    TP = np.sum(tps)
    FP = np.sum(fps)
    FN = np.sum(fns)
    return TP, FP, FN, tps, fps, fns


def calculate_precision_recall_F1score(
    TP: int, FP: int, FN: int
) -> tuple[float, float, float]:
    """
    Calculate precision, recall and F1 score,  from TP, FP and FN numbers
    """
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, F1score


def compute_residuals(
    picks_truth: list[np.ndarray],
    picks_predicted: list[np.ndarray],
    sampling_rate: float,
    method=0,
) -> np.ndarray:
    residuals = []
    if method == 0:
        for i in range(len(picks_truth)):
            if len(picks_truth[i]) > 0 and len(picks_predicted[i]) > 0:
                for j in range(len(picks_predicted[i])):
                    # Compare each predicted pick with each true pick in a trace
                    # There may be more than one true pick in a trace
                    res = picks_predicted[i][j] - picks_truth[i]
                    absmin_index = np.argmin(
                        np.abs(res)
                    )  # use min in case that picks_truth[i] could be a list of true picks
                    residuals.append(res[absmin_index] / sampling_rate)
    elif method == 1:
        for i in range(len(picks_truth)):
            if len(picks_truth[i]) > 0 and len(picks_predicted[i]) > 0:
                assert len(picks_truth[i]) == 1
                res = picks_predicted[i] - picks_truth[i][0]
                absmin_index = np.argmin(
                    np.abs(res)
                )  # use min in case that picks_truth[i] could be a list of true picks
                residuals.append(res[absmin_index] / sampling_rate)
    return np.array(residuals)


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


def eval_task0(
    dev_test_set,
    model,
    exp_name: str,
    targets_path: Path,
    prob_thres: np.ndarray = np.arange(start=0.1, stop=1.0, step=0.1),
    tp_thre: float = 0.5,
    num_workers: int = 12,
    batchsize: int = 1024,
    num_test_run: int = -1,
    output_remark: str = "pred",
    data_norm_type: str = "peak",
    sets=["dev", "test"],
    root_save_dir=None,
    append_to_file=False,
    count_tp_method=0,
    no_p=False,
    no_s=False,
    # search_method="grid",
    # binary_search_n=-1,
):
    """
    dev_test_set: dataset of test and dev examples
    targets_path: parent path of the task0.csv
    num_test_run: for debugging purpose, if it is greater than 0, only the specified number of examples will be evaluated
    search_method: grid (search) or binary (search)
    """
    assert not (no_p and no_s)
    # if search_method not in ["binary", "grid"]:
    #     raise ValueError("search_method should be binary or grid")
    # if search_method == "binary" and binary_search_n < 2:
    #     raise ValueError("binary_search_n should be greater than 1")
    if root_save_dir is None:
        pred_path = (
            targets_path.parent / f"{targets_path.name}_{output_remark}" / "task0"
        )
    else:
        root_save_dir = Path(root_save_dir)
        pred_path = root_save_dir / f"{targets_path.name}_{output_remark}" / "task0"
    save_dir = pred_path / exp_name
    save_dir.mkdir(exist_ok=True, parents=True)
    try:
        save_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{save_dir} exists")
    if not isinstance(sets, Iterable):
        sets = [sets]
    save_score = False
    for eval_set in sets:
        # dataset
        split = dev_test_set.get_split(eval_set)
        if targets_path.name == "instance":
            logging.warning(
                "Overwriting noise trace_names to allow correct identification"
            )
            # Replace trace names for noise entries
            split._metadata["trace_name"].values[-len(split.datasets[-1]) :] = (
                split._metadata["trace_name"][-len(split.datasets[-1]) :].apply(
                    lambda x: "noise_" + x
                )
            )
            split._build_trace_name_to_idx_dict()

        print(f"Starting set: {eval_set}")
        split.preload_waveforms(pbar=True)

        # steered metatdata
        task_csv = targets_path / "task0.csv"
        task_targets = pd.read_csv(task_csv)
        task_targets = task_targets[task_targets["trace_split"] == eval_set]
        if no_p:
            task_targets = task_targets[pd.isna(task_targets["trace_p_arrival_sample"])]
        if no_s:
            task_targets = task_targets[pd.isna(task_targets["trace_s_arrival_sample"])]
        if targets_path.name == "instance":
            border = _identify_instance_dataset_border(task_targets)
            task_targets["trace_name"].values[border:] = task_targets["trace_name"][
                border:
            ].apply(lambda x: "noise_" + x)

        if num_test_run > 0:
            subset_idxs = np.sort(
                np.random.default_rng(seed=100).choice(
                    len(task_targets), num_test_run, replace=False
                )
            )  # sort inplace
            task_targets = task_targets.iloc[subset_idxs]

        generator = sbg.SteeredGenerator(split, task_targets)
        # windowlen = {"PhaseNet": 3001, "EQTransformer": 6000, "VolEQTransformer": 6000}
        windowlen = {"PhaseNet": 3001, "EQTransformer": 6000}
        generator.add_augmentations(
            [
                sbg.SteeredWindow(windowlen=windowlen[model.name], strategy="pad"),
                sbg.ChangeDtype(np.float32),
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=data_norm_type
                ),
            ]
        )
        print(f"Number of examples in {eval_set}: {len(generator)}")
        n_eq_traces = len(
            task_targets[
                (task_targets["trace_split"] == eval_set)
                & (task_targets["trace_type"] == "earthquake")
            ]
        )
        n_noise_traces = len(
            task_targets[
                (task_targets["trace_split"] == eval_set)
                & (task_targets["trace_type"] == "noise")
            ]
        )
        print(f"{n_eq_traces} earthquake traces")
        print(f"{n_noise_traces} noise traces")
        metrics = []
        # for prob_thre in tqdm(np.arange(start=0.1,stop=1.0,step=0.1),desc=f"Probability threshold {prob_thre:.2f}"):
        # if search_method == "grid":
        #     search_maxit = len(prob_thres)
        # elif search_method == "binary":
        #     search_maxit = binary_search_n
        #     p_prob_search_interval = list(prob_thres)
        #     s_prob_search_interval = list(prob_thres)
        #     p_candicate_scores = []
        #     s_candicate_scores = []
        #     assert len(prob_thres) == 2

        for prob_thre in prob_thres:
            # for search_i in range(search_maxit):
            #     if search_method == "grid":
            #         prob_thre = prob_thres[search_i]
            #     elif search_method == "binary":
            #         if search_i < 2:
            #             prob_thre = prob_thres[search_i]
            #         else:
            #             prob_thre = 0.5 * (
            #                 prob_search_interval[0] + prob_search_interval[1]
            #             )
            #             # prob_search_interval
            print(f"Probability threshold: {prob_thre:.4f}")
            # p_idxs_predicted, p_scores, s_idxs_predicted, s_scores
            predictions = evaluate(
                generator=generator,
                model=model,
                threshold=prob_thre,
                batchsize=batchsize,
                num_workers=num_workers,
                save_prob=False,
            )
            p_idxs_predicted = predictions[0]
            p_scores = predictions[1]
            s_idxs_predicted = predictions[2]
            s_scores = predictions[3]
            # if model.name == "VolEQTransformer":
            #     score_rg_detection = predictions[4]
            #     score_lp_detection = predictions[5]
            #     score_lp_over_rg = predictions[6]
            #     task_targets["score_rg_detection"] = score_rg_detection
            #     task_targets["score_lp_detection"] = score_lp_detection
            #     task_targets["score_lp_over_rg"] = score_lp_over_rg

            task_targets[f"p_pred_sample_thr{prob_thre:.4f}"] = p_idxs_predicted
            task_targets[f"s_pred_sample_thr{prob_thre:.4f}"] = s_idxs_predicted
            if save_score:
                task_targets[f"p_pred_score_thr{prob_thre:.4f}"] = p_scores
                task_targets[f"s_pred_score_thr{prob_thre:.4f}"] = s_scores

            if targets_path.name == "instance":
                p_arrival_sample_column = "trace_P_arrival_sample"
                s_arrival_sample_column = "trace_S_arrival_sample"
            else:
                p_arrival_sample_column = "trace_p_arrival_sample"
                s_arrival_sample_column = "trace_s_arrival_sample"
            p_idxs_ground_truth, s_idxs_ground_truth = get_ground_truth(
                generator, p_arrival_sample_column, s_arrival_sample_column
            )

            if not no_p:
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
            if not no_s:
                TP_s, FP_s, FN_s, tps_s, fps_s, fns_s = count_TP_FP_FN(
                    s_idxs_ground_truth,
                    s_idxs_predicted,
                    tp_thre=tp_thre,
                    sampling_rate=100,
                    method=count_tp_method,
                )
                precision_s, recall_s, F1score_s = calculate_precision_recall_F1score(
                    TP_s, FP_s, FN_s
                )
            if not no_p:
                residuals_p = compute_residuals(
                    p_idxs_ground_truth,
                    p_idxs_predicted,
                    sampling_rate=100,
                    method=count_tp_method,
                )
                if len(residuals_p) == 0:
                    print(f"No predicted P picks")
                    p_std = None
                    p_mean = None
                    p_median = None
                    p_mae = None
                    p_mad = None

                    p_outlier = None
                    p_out = None

                    modified_residuals_p = None
                    modified_residuals_p2 = None

                    modified_p_std = None
                    modified_p_rmse = None
                    modified_p_mae = None

                    modified_p_std2 = None
                    modified_p_rmse2 = None
                    modified_p_mae2 = None

                    modified_p_mean = None
                    modified_p_median = None
                    modified_p_mean2 = None
                    modified_p_median2 = None
                    modified_p_mad = None
                    modified_p_mad2 = None
                else:
                    p_std = np.std(residuals_p, ddof=1)
                    p_mean = np.mean(residuals_p)
                    p_median = np.median(residuals_p)
                    p_mae = np.mean(np.abs(residuals_p))  # mean absolute error
                    p_mad = np.median(np.abs(residuals_p - np.median(residuals_p)))

                    p_outlier = residuals_p[
                        (residuals_p < -1) | (residuals_p > 1)
                    ].copy()
                    p_out = p_outlier.size / residuals_p.size

                    modified_residuals_p = residuals_p.copy()  # modified residuals
                    modified_residuals_p[modified_residuals_p < -1] = -1
                    modified_residuals_p[modified_residuals_p > 1] = 1
                    modified_p_std = np.std(modified_residuals_p, ddof=1)
                    modified_p_rmse = np.sqrt(np.mean(modified_residuals_p**2))
                    modified_p_mae = np.mean(np.abs(modified_residuals_p))

                    modified_p_mean = np.mean(modified_residuals_p)
                    modified_p_median = np.median(modified_residuals_p)
                    modified_p_mad = np.median(
                        np.abs(modified_residuals_p - np.median(modified_residuals_p))
                    )

                    modified_residuals_p2 = residuals_p.copy()
                    modified_residuals_p2 = modified_residuals_p2[
                        (modified_residuals_p2 > -1) & (modified_residuals_p2 < 1)
                    ]
                    modified_p_mean2 = np.mean(modified_residuals_p2)
                    modified_p_median2 = np.median(modified_residuals_p2)
                    modified_p_std2 = np.std(modified_residuals_p2, ddof=1)
                    modified_p_rmse2 = np.sqrt(np.mean(modified_residuals_p2**2))
                    modified_p_mae2 = np.mean(np.abs(modified_residuals_p2))

                    modified_p_mad2 = np.median(
                        np.abs(modified_residuals_p2 - np.median(modified_residuals_p2))
                    )
                    # m_p_mad = np.median(
                    #     np.abs(modified_residuals_p - np.median(modified_residuals_p))
                    # )

            if not no_s:
                residuals_s = compute_residuals(
                    s_idxs_ground_truth,
                    s_idxs_predicted,
                    sampling_rate=100,
                    method=count_tp_method,
                )
                if len(residuals_s) == 0:
                    print(f"No predicted S picks")
                    s_std = None
                    s_mean = None
                    s_median = None
                    s_mae = None
                    s_mad = None

                    s_outlier = None
                    s_out = None
                    modified_residuals_s = None
                    modified_s_std = None
                    modified_s_rmse = None
                    modified_s_mae = None

                    modified_residuals_s2 = None
                    modified_s_std2 = None
                    modified_s_rmse2 = None
                    modified_s_mae2 = None

                    modified_s_mean = None
                    modified_s_median = None
                    modified_s_mean2 = None
                    modified_s_median2 = None
                    modified_s_mad = None
                    modified_s_mad2 = None
                else:
                    s_std = np.std(residuals_s, ddof=1)
                    s_mean = np.mean(residuals_s)
                    s_median = np.median(residuals_s)
                    s_mae = np.mean(np.abs(residuals_s))
                    s_mad = np.median(np.abs(residuals_s - np.median(residuals_s)))

                    s_outlier = residuals_s[
                        (residuals_s < -1) | (residuals_s > 1)
                    ].copy()
                    s_out = s_outlier.size / residuals_s.size
                    modified_residuals_s = residuals_s.copy()
                    modified_residuals_s[modified_residuals_s < -1] = -1
                    modified_residuals_s[modified_residuals_s > 1] = 1
                    modified_s_std = np.std(modified_residuals_s, ddof=1)
                    modified_s_rmse = np.sqrt(np.mean(modified_residuals_s**2))
                    modified_s_mae = np.mean(np.abs(modified_residuals_s))

                    modified_s_mean = np.mean(modified_residuals_s)
                    modified_s_median = np.median(modified_residuals_s)
                    modified_s_mad = np.median(
                        np.abs(modified_residuals_s - np.median(modified_residuals_s))
                    )

                    modified_residuals_s2 = residuals_s.copy()
                    modified_residuals_s2 = modified_residuals_s2[
                        (modified_residuals_s2 > -1) & (modified_residuals_s2 < 1)
                    ]

                    modified_s_std2 = np.std(modified_residuals_s2, ddof=1)
                    modified_s_rmse2 = np.sqrt(np.mean(modified_residuals_s2**2))
                    modified_s_mae2 = np.mean(np.abs(modified_residuals_s2))

                    modified_s_mean2 = np.mean(modified_residuals_s2)
                    modified_s_median2 = np.median(modified_residuals_s2)

                    modified_s_mad2 = np.median(
                        np.abs(modified_residuals_s2 - np.median(modified_residuals_s2))
                    )
                    # m_s_mad = np.median(
                    #     np.abs(modified_residuals_s - np.median(modified_residuals_s))
                    # )
            if not no_p:
                p_stats = {
                    "p_TP": TP_p,
                    "p_FP": FP_p,
                    "p_FN": FN_p,
                    "p_precision": precision_p,
                    "p_recall": recall_p,
                    "p_F1score": F1score_p,
                    "p_mean": p_mean,
                    "p_median": p_median,
                    "p_std": p_std,
                    "p_MAE": p_mae,
                    "p_MAD": p_mad,
                    "p_out": p_out,
                    "p_modified_mean": modified_p_mean,
                    "p_modified_median": modified_p_median,
                    "p_modified_std": modified_p_std,
                    "p_modified_RMSE": modified_p_rmse,
                    "p_modified_MAE": modified_p_mae,
                    "p_modified_MAD": modified_p_mad,
                    "p_modified_mean2": modified_p_mean2,
                    "p_modified_median2": modified_p_median2,
                    "p_modified_std2": modified_p_std2,
                    "p_modified_RMSE2": modified_p_rmse2,
                    "p_modified_MAE2": modified_p_mae2,
                    "p_modified_MAD2": modified_p_mad2,
                }
            else:
                p_stats = {}
            if not no_s:
                s_stats = {
                    "s_TP": TP_s,
                    "s_FP": FP_s,
                    "s_FN": FN_s,
                    "s_precision": precision_s,
                    "s_recall": recall_s,
                    "s_F1score": F1score_s,
                    "s_mean": s_mean,
                    "s_median": s_median,
                    "s_std": s_std,
                    "s_MAE": s_mae,
                    "s_MAD": s_mad,
                    "s_out": s_out,
                    "s_modified_mean": modified_s_mean,
                    "s_modified_median": modified_s_median,
                    "s_modified_std": modified_s_std,
                    "s_modified_RMSE": modified_s_rmse,
                    "s_modified_MAE": modified_s_mae,
                    "s_modified_MAD": modified_s_mad,
                    "s_modified_mean2": modified_s_mean2,
                    "s_modified_median2": modified_s_median2,
                    "s_modified_std2": modified_s_std2,
                    "s_modified_RMSE2": modified_s_rmse2,
                    "s_modified_MAE2": modified_s_mae2,
                    "s_modified_MAD2": modified_s_mad2,
                }
            else:
                s_stats = {}

            metrics.append(
                {"prob_thre": prob_thre, "tp_thre": tp_thre, **p_stats, **s_stats}
            )
        metrics_form = pd.DataFrame(metrics)

        # task_targets.to_csv(, index=False)
        if no_s:
            fname_suffix = "_p"
        if no_p:
            fname_suffix = "_s"
        if (not no_p) and (not no_s):
            fname_suffix = ""
        print(fname_suffix)

        output_metrics_file = (
            pred_path / exp_name / f"{eval_set}_metrics{fname_suffix}.csv"
        )
        if append_to_file == True and output_metrics_file.exists():
            previous_metrics_form = pd.read_csv(output_metrics_file, index_col=False)
            new_metrics_form = pd.concat(
                [previous_metrics_form, metrics_form], ignore_index=True
            )
            new_metrics_form.sort_values(
                by=["tp_thre", "prob_thre"], ignore_index=True, inplace=True
            )
            new_metrics_form.to_csv(output_metrics_file, index=False)
            previous_task_targets = pd.read_csv(
                pred_path / exp_name / f"{eval_set}_task0{fname_suffix}.csv",
                index_col=False,
            )
            for col_name in task_targets.columns:
                if (col_name not in previous_task_targets) and (
                    col_name.startswith("p_pred_sample")
                    or col_name.startswith("s_pred_sample")
                ):
                    previous_task_targets[col_name] = task_targets[col_name]
            previous_task_targets.to_csv(
                pred_path / exp_name / f"{eval_set}_task0{fname_suffix}.csv",
                index=False,
            )
        else:
            metrics_form.to_csv(output_metrics_file, index=False)
            task_targets.to_csv(
                pred_path / exp_name / f"{eval_set}_task0{fname_suffix}.csv",
                index=False,
            )


def eval_task0_true_negative_rate(
    dev_test_set,
    model,
    exp_name: str,
    targets_path: Path,
    prob_thres: np.ndarray = np.arange(start=0.1, stop=1.0, step=0.1),
    num_workers: int = 12,
    batchsize: int = 1024,
    num_test_run: int = -1,
    output_remark: str = "pred",
    data_norm_type: str = "peak",
    sets=["dev", "test"],
    root_save_dir=None,
    append_to_file=False,
):
    """
    dev_test_set: dataset of test and dev examples
    targets_path: parent path of the task0.csv
    num_test_run: for debugging purpose, if it is greater than 0, only the specified number of examples will be evaluated
    """
    if root_save_dir is None:
        pred_path = (
            targets_path.parent / f"{targets_path.name}_{output_remark}" / "task0"
        )
    else:
        root_save_dir = Path(root_save_dir)
        pred_path = root_save_dir / f"{targets_path.name}_{output_remark}" / "task0"
    save_dir = pred_path / exp_name
    save_dir.mkdir(exist_ok=True, parents=True)
    try:
        save_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{save_dir} exists")
    if not isinstance(sets, Iterable):
        sets = [sets]
    save_score = False
    for eval_set in sets:
        # dataset
        split = dev_test_set.get_split(eval_set)
        print(f"Starting set: {eval_set}")
        split.preload_waveforms(pbar=True)

        # steered metatdata
        task_csv = targets_path / "task0.csv"
        task_targets = pd.read_csv(task_csv)
        task_targets = task_targets[task_targets["trace_split"] == eval_set]

        if num_test_run > 0:
            subset_idxs = np.sort(
                np.random.default_rng(seed=100).choice(
                    len(task_targets), num_test_run, replace=False
                )
            )  # sort inplace
            task_targets = task_targets.iloc[subset_idxs]

        generator = sbg.SteeredGenerator(split, task_targets)
        # windowlen = {"PhaseNet": 3001, "EQTransformer": 6000, "VolEQTransformer": 6000}
        windowlen = {"PhaseNet": 3001, "EQTransformer": 6000}
        generator.add_augmentations(
            [
                sbg.SteeredWindow(windowlen=windowlen[model.name], strategy="pad"),
                sbg.ChangeDtype(np.float32),
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=data_norm_type
                ),
            ]
        )
        print(f"Number of examples in {eval_set}: {len(generator)}")
        n_eq_traces = len(
            task_targets[
                (task_targets["trace_split"] == eval_set)
                & (task_targets["trace_type"] == "earthquake")
            ]
        )
        n_noise_traces = len(
            task_targets[
                (task_targets["trace_split"] == eval_set)
                & (task_targets["trace_type"] == "noise")
            ]
        )
        print(f"{n_eq_traces} earthquake traces")
        print(f"{n_noise_traces} noise traces")
        metrics = []
        # for prob_thre in tqdm(np.arange(start=0.1,stop=1.0,step=0.1),desc=f"Probability threshold {prob_thre:.2f}"):
        for prob_thre in prob_thres:
            print(f"Probability threshold: {prob_thre:.3f}")
            # p_idxs_predicted, p_scores, s_idxs_predicted, s_scores
            predictions = evaluate(
                generator=generator,
                model=model,
                threshold=prob_thre,
                batchsize=batchsize,
                num_workers=num_workers,
                save_prob=False,
            )
            p_idxs_predicted = predictions[0]
            p_scores = predictions[1]
            s_idxs_predicted = predictions[2]
            s_scores = predictions[3]
            # if model.name == "VolEQTransformer":
            #     score_rg_detection = predictions[4]
            #     score_lp_detection = predictions[5]
            #     score_lp_over_rg = predictions[6]
            #     task_targets["score_rg_detection"] = score_rg_detection
            #     task_targets["score_lp_detection"] = score_lp_detection
            #     task_targets["score_lp_over_rg"] = score_lp_over_rg

            task_targets[f"p_pred_sample_thr{prob_thre:.3f}"] = p_idxs_predicted
            task_targets[f"s_pred_sample_thr{prob_thre:.3f}"] = s_idxs_predicted
            if save_score:
                task_targets[f"p_pred_score_thr{prob_thre:.3f}"] = p_scores
                task_targets[f"s_pred_score_thr{prob_thre:.3f}"] = s_scores

            p_idxs_ground_truth, s_idxs_ground_truth = get_ground_truth(generator)

            def count_TN_FP(
                picks_truth, picks_predicted
            ):  # different from count_TP_FP_FN in counting the number of FP
                TN = 0
                FP = 0
                for i in range(len(picks_predicted)):
                    if len(picks_predicted[i]) == 0 and len(picks_truth[i]) == 0:
                        TN += 1
                    elif len(picks_predicted[i]) > 0 and len(picks_truth[i]) == 0:
                        FP += 1
                return TN, FP

            TN_p, FP_p = count_TN_FP(p_idxs_ground_truth, p_idxs_predicted)
            TN_s, FP_s = count_TN_FP(s_idxs_ground_truth, s_idxs_predicted)

            p_tnr = TN_p / (TN_p + FP_p)
            s_tnr = TN_s / (TN_s + FP_s)

            metrics.append(
                {
                    "prob_thre": prob_thre,
                    "p_TN": TN_p,
                    "p_FP": FP_p,
                    "p_true_negative_rate": p_tnr,
                    "s_TN": TN_s,
                    "s_FP": FP_s,
                    "s_true_negative_rate": s_tnr,
                }
            )
        metrics_form = pd.DataFrame(metrics)

        # task_targets.to_csv(, index=False)

        output_metrics_file = pred_path / exp_name / f"{eval_set}_tnr.csv"
        metrics_form.to_csv(output_metrics_file, index=False)
        task_targets.to_csv(
            pred_path / exp_name / f"{eval_set}_task0_negatives.csv", index=False
        )


def parse_task0_detection(exp_dir: Path, only_dev=True):
    if not (
        (exp_dir / "dev_task0.csv").is_file() and (exp_dir / "test_task0.csv").is_file()
    ):
        logging.warning(f"Directory {exp_dir} does not contain task 0")
        return {}

    stats = {}

    dev_pred = pd.read_csv(exp_dir / "dev_task0.csv")
    dev_pred["source_type_lp_bin"] = dev_pred["source_type"] == "lp"
    dev_pred["source_type_rg_bin"] = (dev_pred["source_type"] != "lp") & (
        dev_pred["source_type"] != "noise"
    )
    # dev set
    # lp detection
    if dev_pred["source_type_lp_bin"].any():
        lp_prec, lp_recall, lp_thr = precision_recall_curve(
            dev_pred["source_type_lp_bin"], dev_pred["score_lp_detection"]
        )
        lp_f1 = 2 * lp_prec * lp_recall / (lp_prec + lp_recall)
        lp_auc = roc_auc_score(
            dev_pred["source_type_lp_bin"], dev_pred["score_lp_detection"]
        )
        lp_opt_index = np.nanargmax(lp_f1)  # F1 optimal threshold index
        lp_opt_thr = lp_thr[lp_opt_index]  # F1 optimal threshold value

        stats.update(
            {
                "dev_lp_det_precision": lp_prec[lp_opt_index],
                "dev_lp_det_recall": lp_recall[lp_opt_index],
                "dev_lp_det_f1": lp_f1[lp_opt_index],
                "dev_lp_det_auc": lp_auc,
                "det_lp_threshold": lp_opt_thr,
            }
        )
    if dev_pred["source_type_rg_bin"].any():
        rg_prec, rg_recall, rg_thr = precision_recall_curve(
            dev_pred["source_type_rg_bin"], dev_pred["score_rg_detection"]
        )
        rg_f1 = 2 * rg_prec * rg_recall / (rg_prec + rg_recall)
        rg_auc = roc_auc_score(
            dev_pred["source_type_rg_bin"], dev_pred["score_rg_detection"]
        )
        rg_opt_index = np.nanargmax(rg_f1)  # F1 optimal threshold index
        rg_opt_thr = rg_thr[rg_opt_index]  # F1 optimal threshold value
        stats.update(
            {
                "dev_rg_det_precision": rg_prec[rg_opt_index],
                "dev_rg_det_recall": rg_recall[rg_opt_index],
                "dev_rg_det_f1": rg_f1[rg_opt_index],
                "dev_rg_det_auc": rg_auc,
                "det_rg_threshold": rg_opt_thr,
            }
        )

    if not only_dev:
        # test set
        test_pred = pd.read_csv(exp_dir / "test_task0.csv")
        test_pred["source_type_lp_bin"] = test_pred["source_type"] == "lp"
        test_pred["source_type_rg_bin"] = (test_pred["source_type"] != "lp") & (
            test_pred["source_type"] != "noise"
        )
        if test_pred["source_type_lp_bin"].any() and "det_lp_threshold" in stats:
            lp_prec, lp_recall, lp_f1, _ = precision_recall_fscore_support(
                test_pred["source_type_lp_bin"],
                test_pred["score_lp_detection"] > lp_opt_thr,
                average="binary",
            )
            lp_auc = roc_auc_score(
                test_pred["source_type_lp_bin"], test_pred["score_lp_detection"]
            )
            stats.update(
                {
                    "test_lp_det_precision": lp_prec,
                    "test_lp_det_recall": lp_recall,
                    "test_lp_det_f1": lp_f1,
                    "test_lp_det_auc": lp_auc,
                }
            )

        if test_pred["source_type_rg_bin"].any() and "det_rg_threshold" in stats:
            rg_prec, rg_recall, rg_f1, _ = precision_recall_fscore_support(
                test_pred["source_type_rg_bin"],
                test_pred["score_rg_detection"] > rg_opt_thr,
                average="binary",
            )
            rg_auc = roc_auc_score(
                test_pred["source_type_rg_bin"], test_pred["score_rg_detection"]
            )
            stats.update(
                {
                    "test_rg_det_precision": rg_prec,
                    "test_rg_det_recall": rg_recall,
                    "test_rg_det_f1": rg_f1,
                    "test_rg_det_auc": rg_auc,
                }
            )
        # test_stats = {
        #     "test_lp_det_precision": lp_prec,
        #     "test_lp_det_recall": lp_recall,
        #     "test_lp_det_f1": lp_f1,
        #     "test_lp_det_auc": lp_auc,
        #     "test_rg_det_precision": rg_prec,
        #     "test_rg_det_recall": rg_recall,
        #     "test_rg_det_f1": rg_f1,
        #     "test_rg_det_auc": rg_auc,
        # }
        # stats.update(test_stats)

    # source type identification
    dev_pred = dev_pred[dev_pred["source_type"] != "noise"]

    # print(np.unique(dev_pred["source_type"]))
    # dev_pred["score_lp_over_rg"]=dev_pred["score_lp_detection"]/dev_pred["score_rg_detection"]
    # test_pred["score_lp_over_rg"]=test_pred["score_lp_detection"]/test_pred["score_rg_detection"]
    # print(dev_pred[["source_type","score_lp_over_rg"]])
    dev_pred["lp_or_rg_bin"] = dev_pred["source_type"] == "lp"

    print(np.min(dev_pred["score_lp_over_rg"]), np.max(dev_pred["score_lp_over_rg"]))
    dev_pred["score_lp_over_rg"] = np.clip(
        dev_pred["score_lp_over_rg"].values, -1e100, 1e100
    )

    # Determine (approximately) optimal MCC threshold using 50 candidates
    mcc_thrs = np.sort(dev_pred["score_lp_over_rg"].values)
    mcc_thrs = mcc_thrs[np.linspace(0, len(mcc_thrs) - 1, 50, dtype=int)]
    dev_mccs = []
    for thr in mcc_thrs:
        dev_mccs.append(
            matthews_corrcoef(
                dev_pred["lp_or_rg_bin"], dev_pred["score_lp_over_rg"] > thr
            )
        )
    dev_mcc = np.max(dev_mccs)
    mcc_thr = mcc_thrs[np.argmax(dev_mccs)]
    dev_stats = {
        "dev_source_type_mcc": dev_mcc,
        "source_type_threshold_mcc": mcc_thr,
    }
    stats.update(dev_stats)

    if not only_dev:
        # mcc for the test set
        test_pred = test_pred[test_pred["source_type"] != "noise"]
        test_pred["lp_or_rg_bin"] = test_pred["source_type"] == "lp"
        test_pred["score_lp_over_rg"] = np.clip(
            test_pred["score_lp_over_rg"].values, -1e100, 1e100
        )
        test_mcc = matthews_corrcoef(
            test_pred["lp_or_rg_bin"], test_pred["score_lp_over_rg"] > mcc_thr
        )
        stats.update({"test_source_type_mcc": test_mcc})
    return stats


def opt_prob_metrics(exp_dir: Path) -> dict:
    test_metrics = pd.read_csv(exp_dir / "test_metrics.csv")
    dev_metrics = pd.read_csv(exp_dir / "dev_metrics.csv")
    p_opt_row_id = np.argmax(dev_metrics["p_F1score"])
    p_opt_thre = dev_metrics.iloc[p_opt_row_id]["prob_thre"]
    s_opt_row_id = np.argmax(dev_metrics["s_F1score"])
    s_opt_thre = dev_metrics.iloc[s_opt_row_id]["prob_thre"]

    opt_p_test = test_metrics.iloc[p_opt_row_id][
        [x for x in list(test_metrics.columns) if x[0:2] == "p_"]
    ].to_dict()
    opt_p_dev = dev_metrics.iloc[p_opt_row_id][
        [x for x in list(dev_metrics.columns) if x[0:2] == "p_"]
    ].to_dict()
    opt_s_test = test_metrics.iloc[s_opt_row_id][
        [x for x in list(test_metrics.columns) if x[0:2] == "s_"]
    ].to_dict()
    opt_s_dev = dev_metrics.iloc[s_opt_row_id][
        [x for x in list(dev_metrics.columns) if x[0:2] == "s_"]
    ].to_dict()
    result = {}
    result["exp_name"] = exp_dir.name
    result["tp_thre"] = test_metrics.iloc[0]["tp_thre"]
    result["p_opt_prob_thre"] = p_opt_thre
    result["s_opt_prob_thre"] = s_opt_thre
    for key, value in opt_p_test.items():
        result["test_" + key] = value
    for key, value in opt_s_test.items():
        result["test_" + key] = value
    for key, value in opt_p_dev.items():
        result["dev_" + key] = value
    for key, value in opt_s_dev.items():
        result["dev_" + key] = value
    return result


def collect_task0_results(
    pred_dir: Path,
    output_fname="task0_metrics.csv",
    parse_exp_name=True,
    # lp_detection=False,
):
    exp_names = [x for x in pred_dir.iterdir() if x.is_dir()]  # and x.name != "task123"
    print(exp_names)
    metrics_data = []
    for exp_dir in exp_names:
        stats = opt_prob_metrics(exp_dir)
        # if lp_detection:
        #     det_stats = parse_task0_detection(exp_dir, only_dev=False)
        #     stats.update(det_stats)
        metrics_data.append(stats)

    df = pd.DataFrame(metrics_data)
    df.sort_values(by=["tp_thre"], ignore_index=True, inplace=True)
    if parse_exp_name:
        # model_names = {"ve": "voleqtransformer", "e": "eqtransformer", "p": "phasenet"}
        model_names = {"e": "eqtransformer", "p": "phasenet"}
        label_function = {"ga": "gaussian", "tr": "triangle"}
        df.insert(
            1, "model", df["exp_name"].apply(lambda x: model_names[x.split("_")[0]])
        )
        df.insert(2, "batch_size", df["exp_name"].apply(lambda x: x.split("_")[1]))
        df.insert(3, "lr", df["exp_name"].apply(lambda x: x.split("_")[2]))
        df.insert(
            4,
            "label_function",
            df["exp_name"].apply(lambda x: label_function[x.split("_")[3][:2]]),
        )
        df.insert(
            5,
            "label_var",
            df["exp_name"].apply(lambda x: x.split("_")[3][2:]),
        )
        df.insert(6, "max_epoch", df["exp_name"].apply(lambda x: x.split("_")[4]))
    df.to_csv(pred_dir / output_fname, index=False)


def opt_prob_metrics_dev(exp_dir: Path) -> dict:
    dev_metrics = pd.read_csv(exp_dir / "dev_metrics.csv")
    p_opt_row_id = np.argmax(dev_metrics["p_F1score"])
    p_opt_thre = dev_metrics.iloc[p_opt_row_id]["prob_thre"]
    s_opt_row = np.argmax(dev_metrics["s_F1score"])
    s_opt_thre = dev_metrics.iloc[s_opt_row]["prob_thre"]

    opt_p_dev = dev_metrics.iloc[p_opt_row_id][
        [x for x in list(dev_metrics.columns) if x[0:2] == "p_"]
    ].to_dict()
    opt_s_dev = dev_metrics.iloc[s_opt_row][
        [x for x in list(dev_metrics.columns) if x[0:2] == "s_"]
    ].to_dict()
    result = {}
    result["exp_name"] = exp_dir.name
    result["tp_thre"] = dev_metrics.iloc[0]["tp_thre"]
    result["p_opt_prob_thre"] = p_opt_thre
    result["s_opt_prob_thre"] = s_opt_thre
    for key, value in opt_p_dev.items():
        result["dev_" + key] = value
    for key, value in opt_s_dev.items():
        result["dev_" + key] = value
    return result


def collect_task0_results_dev(
    pred_dir: Path,
    output_fname: str = "task0_metrics_sum.csv",
    parse_exp_name: bool = True,
) -> pd.DataFrame:
    exp_names = [
        x
        for x in pred_dir.iterdir()
        if x.is_dir() and x.name != "task123" and (x / "dev_metrics.csv").exists()
    ]
    # print(exp_names)
    metrics_data = []
    for exp_dir in exp_names:
        print(exp_dir.name)
        stats = opt_prob_metrics_dev(exp_dir)
        # det_stats = parse_task0_detection(exp_dir, only_dev=True)
        # stats.update(det_stats)
        metrics_data.append(stats)

    df = pd.DataFrame(metrics_data)
    df.sort_values(by=["tp_thre"], ignore_index=True, inplace=True)

    if parse_exp_name:
        # model_names = {"ve": "voleqtransformer", "e": "eqtransformer", "p": "phasenet"}
        model_names = {"e": "eqtransformer", "p": "phasenet"}
        label_function = {"ga": "gaussian", "tr": "triangle"}
        df.insert(
            1, "model", df["exp_name"].apply(lambda x: model_names[x.split("_")[0]])
        )
        df.insert(2, "batch_size", df["exp_name"].apply(lambda x: x.split("_")[1]))
        df.insert(3, "lr", df["exp_name"].apply(lambda x: x.split("_")[2]))
        df.insert(
            4,
            "label_function",
            df["exp_name"].apply(lambda x: label_function[x.split("_")[3][:2]]),
        )
        df.insert(
            5,
            "label_var",
            df["exp_name"].apply(lambda x: x.split("_")[3][2:]),
        )
        df.insert(6, "max_epoch", df["exp_name"].apply(lambda x: x.split("_")[4]))

        df.insert(
            7,
            "pre-trained_on",
            df["exp_name"].apply(
                lambda x: x.split("_")[6][3:] if len(x.split("_")) > 6 else "None"
            ),
        )

    df.sort_values(by=["model"], ignore_index=True, inplace=True)
    df.to_csv(pred_dir / output_fname, index=False)
    return df


def get_optimal_model(df):
    x = df[["dev_det_auc", "dev_phase_mcc", "dev_P_std_s", "dev_S_std_s"]].values.copy()
    x[:, 2:] = 1 / x[:, 2:]
    x /= np.max(x, axis=0, keepdims=True)
    means = np.nanmean(x, axis=1)
    if np.isnan(means).all():
        return None

    return np.nanargmax(means)
