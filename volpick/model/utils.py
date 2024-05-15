"""
This script offers general functionality required in multiple places.
"""

import numpy as np
import pandas as pd
import os
import logging
import packaging
import pytorch_lightning as pl
from pathlib import Path
import yaml
import seisbench.generate as sbg
import volpick.model.models as models
from volpick.model.models import phase_dict  # , EventTypeDetectionLabeller
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from volpick.model.eval_taks0 import (
    evaluate,
    get_ground_truth,
    count_TP_FP_FN,
)
from obspy.imaging.spectrogram import spectrogram


def load_best_model_from_base_dir(
    weight_path,
    version=-1,
    model_labels=None,
    plot_loss=True,
    device="cuda",
    return_config=False,
):
    weights = Path(weight_path)
    version = sorted(list(weights.iterdir()), key=lambda x: int(x.name.split("_")[1]))[
        version
    ]
    config_path = version / "hparams.yaml"
    with open(config_path, "r") as f:
        config = yaml.full_load(f)

    model_cls = models.__getattribute__(config["model"] + "Lit")
    lightning_model, best_epoch, best_step = load_best_model(
        model_cls, weights, version.name, return_best_epoch_number=True
    )
    # model=model.model
    lightning_model.model.cuda(device)
    if model_labels is not None:
        lightning_model.model.labels = model_labels
        print(f"Setting model.labels to {model_labels}")
    elif config["model"] == "PhaseNet":
        lightning_model.model.labels = "PSN"
    # model.labels="PSN" # important for evaluate() function to treat the output correctly, because the training data is in the order of P S N !!!

    if plot_loss == True:
        # loss curve
        # print(weights)
        # print(version.name)
        # print((weights / version.name / "metrics.csv"))
        print((weights / version.name / "metrics.csv").exists())
        metrics_log = pd.read_csv(weights / version.name / "metrics.csv")

        # read data
        # epoch_val=metrics_log["epoch"][~pd.isna(metrics_log["val_loss"])]
        # step_val=metrics_log["step"][~pd.isna(metrics_log["val_loss"])]
        # val_loss=metrics_log["val_loss"][~pd.isna(metrics_log["val_loss"])]

        val_loss = metrics_log.drop_duplicates(subset=["epoch"], keep="last")
        train_loss = metrics_log[pd.notna(metrics_log["train_loss"])].drop_duplicates(
            subset=["epoch"], keep="last"
        )
        lrs = metrics_log[pd.notna(metrics_log["lr-Adam"])]["lr-Adam"].to_numpy()
        # plot
        # metrics_epoch_train=metrics_log[~pd.isna(metrics_log["train_loss"])].groupby(["epoch"],as_index=False).last()
        # print(metrics_epoch_train)
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex="col")
        # plt.semilogy(metrics_epoch_train["epoch"]+1,metrics_epoch_train["train_loss"],label="train_loss")
        # plt.semilogy(epoch_val+1,val_loss,label="val_loss")
        axs[0].semilogy(
            train_loss["epoch"] + 1, train_loss["train_loss"], label="Training loss"
        )
        axs[0].semilogy(
            val_loss["epoch"] + 1, val_loss["val_loss"], label="Validation loss"
        )
        # axs[0].axvline()
        axs[0].legend()

        # ax2 = plt.gca().twinx()
        # ax2.spines["right"].set_color("red")
        # ax2.tick_params(axis="y", color="red", labelcolor="red")
        axs[1].semilogy(
            train_loss["epoch"] + 1,
            lrs,
            linestyle="-",
            color="red",
            label="Learning rate",
        )
        axs[1].legend()
        # axs[0].set_xlim([0, 25])
        print(min(val_loss["val_loss"]))
        plt.xlabel("Epoch")
        plt.savefig(weights / version.name / "loss", bbox_inches="tight", dpi=300)
        print(weights / version.name)

    if return_config:
        return lightning_model, config
    else:
        return lightning_model


def load_best_model(model_cls, weights, version, return_best_epoch_number=False):
    """
    Determines the model with lowest validation loss from the csv logs and loads it

    :param model_cls: Class of the lightning module to load
    :param weights: Path to weights as in cmd arguments
    :param version: String of version file
    :return: Instance of lightning module that was loaded from the best checkpoint
    """
    metrics = pd.read_csv(weights / version / "metrics.csv")

    idx = np.nanargmin(metrics["val_loss"])
    min_row = metrics.iloc[idx]

    #  For default checkpoint filename, see https://github.com/Lightning-AI/lightning/pull/11805
    #  and https://github.com/Lightning-AI/lightning/issues/16636.
    #  For example, 'epoch=0-step=1.ckpt' means the 1st step has finish, but the 1st epoch hasn't
    checkpoint = f"epoch={min_row['epoch']:.0f}-step={min_row['step']+1:.0f}.ckpt"

    # For default save path of checkpoints, see https://github.com/Lightning-AI/lightning/pull/12372
    checkpoint_path = weights / version / "checkpoints" / checkpoint

    if return_best_epoch_number:
        return (
            model_cls.load_from_checkpoint(checkpoint_path),
            min_row["epoch"],
            min_row["step"] + 1,
        )
    else:
        return model_cls.load_from_checkpoint(checkpoint_path)


def plot_prediction_examples(
    model,
    targets_path,
    dataset_plot,
    num_test_run,
    sigma,
    fig_dir,
    prob_thre=0.3,
    tp_thre=0.5,
    count_tp_method=0,
    only_false_examples=False,
    only_false_P_examples=False,
    only_false_S_examples=False,
    num_figs=None,
    plot_spectrogram=False,
):
    task_targets = pd.read_csv(targets_path)
    if num_test_run is not None:
        if num_test_run > 0:
            subset_idxs = np.sort(
                np.random.default_rng(seed=100).choice(
                    len(task_targets), num_test_run, replace=False
                )
            )  # sort inplace
            task_targets = task_targets.iloc[subset_idxs]

    # steered generator
    generator = sbg.SteeredGenerator(dataset_plot, task_targets)
    if model.name == "PhaseNet":
        data_norm_type = model.norm
        generator.add_augmentations(
            [
                sbg.SteeredWindow(windowlen=3001, strategy="pad"),
                sbg.ChangeDtype(np.float32),
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=data_norm_type
                ),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=sigma, dim=0),
            ]
        )
    elif model.name == "EQTransformer":
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]
        detection_labeller = sbg.DetectionLabeller(
            p_phases, s_phases=s_phases, key=("X", "detections")
        )
        data_norm_type = model.norm
        generator.add_augmentations(
            [
                sbg.SteeredWindow(windowlen=6000, strategy="pad"),
                sbg.ProbabilisticLabeller(
                    label_columns=phase_dict, sigma=sigma, dim=0, noise_column=False
                ),
                detection_labeller,
                # Normalize to ensure correct augmentation behavior
                sbg.Normalize(
                    detrend_axis=-1, amp_norm_axis=-1, amp_norm_type=data_norm_type
                ),  # "peak"/“std
                sbg.ChangeDtype(np.float32, "X"),
                sbg.ChangeDtype(np.float32, "y"),
                sbg.ChangeDtype(np.float32, "detections"),
            ]
        )
    # elif model.name == "VolEQTransformer":
    #     p_phases = [key for key, val in phase_dict.items() if val == "P"]
    #     s_phases = [key for key, val in phase_dict.items() if val == "S"]
    #     # lp_detection_labeller = sbg.DetectionLabeller(
    #     #     p_phases, s_phases=s_phases, key=("X", "detections")
    #     # )
    #     lp_detection_labeller = EventTypeDetectionLabeller(
    #         source_type="lp",
    #         p_phases=p_phases,
    #         s_phases=s_phases,
    #         key=("X", "lp_detections"),
    #         sigma=20,
    #     )
    #     rg_detection_labeller = EventTypeDetectionLabeller(
    #         source_type="regular",
    #         p_phases=p_phases,
    #         s_phases=s_phases,
    #         key=("X", "rg_detections"),
    #         sigma=20,
    #     )
    #     data_norm_type = model.norm
    #     generator.add_augmentations(
    #         [
    #             sbg.SteeredWindow(windowlen=6000, strategy="pad"),
    #             sbg.ProbabilisticLabeller(
    #                 label_columns=phase_dict, sigma=sigma, dim=0, noise_column=False
    #             ),
    #             rg_detection_labeller,
    #             lp_detection_labeller,
    #             # Normalize to ensure correct augmentation behavior
    #             sbg.Normalize(
    #                 detrend_axis=-1, amp_norm_axis=-1, amp_norm_type=data_norm_type
    #             ),  # "peak"/“std
    #             sbg.ChangeDtype(np.float32, "X"),
    #             sbg.ChangeDtype(np.float32, "y"),
    #             sbg.ChangeDtype(np.float32, "rg_detections"),
    #             sbg.ChangeDtype(np.float32, "lp_detections"),
    #         ]
    #     )

    print(f"Number of examples: {len(generator)}")

    *predictions, probs = evaluate(
        generator=generator,
        model=model,
        threshold=prob_thre,
        batchsize=5,
        num_workers=5,
        save_prob=True,
        show_pbar=True,
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
    p_idxs_ground_truth, s_idxs_ground_truth = get_ground_truth(generator)
    TP_p, FP_p, FN_p, tps_p, fps_p, fns_p = count_TP_FP_FN(
        p_idxs_ground_truth,
        p_idxs_predicted,
        tp_thre=tp_thre,
        sampling_rate=100,
        method=count_tp_method,
    )

    TP_s, FP_s, FN_s, tps_s, fps_s, fns_s = count_TP_FP_FN(
        s_idxs_ground_truth,
        s_idxs_predicted,
        tp_thre=tp_thre,
        sampling_rate=100,
        method=count_tp_method,
    )

    lw = 1
    sampling_rate = 100

    figs_count = 0
    indices = np.arange(len(p_idxs_predicted))
    np.random.shuffle(indices)
    for idx in indices:
        if only_false_examples:
            if not (
                (fps_p[idx] > 0)
                or (fns_p[idx] > 0)
                or (fps_s[idx] > 0)
                or (fns_s[idx] > 0)
            ):
                continue
        if only_false_P_examples:
            if not ((fps_p[idx] > 0) or (fns_p[idx] > 0)):
                continue
        if only_false_S_examples:
            if not (fps_s[idx] > 0) or (fns_s[idx] > 0):
                continue
        if num_figs is not None:
            if figs_count < num_figs:
                figs_count = figs_count + 1
            else:
                break
        # print(idx)

        # for idx in range(0,5,1):
        sample = generator[idx]
        start_sample = generator.metadata.iloc[idx]["start_sample"]
        end_sample = generator.metadata.iloc[idx]["end_sample"]
        local_start_sample, local_end_sample = generator[idx][
            "window_borders"
        ]  # see window.py->SteeredWindow::__call__

        spectrogram_ax = 0
        if plot_spectrogram:
            spectrogram_ax = 3
        # if model.name == "VolEQTransformer":
        #     fig, axs = plt.subplots(
        #         3 + spectrogram_ax,
        #         1,
        #         figsize=(7.5, (3 + spectrogram_ax) * 2),
        #         sharex="col",
        #     )
        # else:
        fig, axs = plt.subplots(
            2 + spectrogram_ax,
            1,
            figsize=(7.5, (2 + spectrogram_ax) * 2),
            sharex="col",
        )
        plt.subplots_adjust(hspace=0.05)
        clrs = ["black", "darkseagreen", "lightskyblue"]
        labels = ["P", "S", "Noise"]
        wave_labs = ["Z", "N", "E"]
        # plot waveforms
        for j in reversed(range(3)):
            local_sample = sample["X"][j][local_start_sample:local_end_sample]
            axs[0].plot(
                (np.arange(len(local_sample))) / sampling_rate,
                local_sample,
                label=wave_labs[j],
                color=clrs[j],
                linewidth=lw,
            )
            if plot_spectrogram:
                spectrogram(
                    data=local_sample,
                    samp_rate=sampling_rate,
                    wlen=256 / sampling_rate,
                    axes=axs[-3 + j],
                    cmap="jet",
                    log=True
                    # dbscale=True,
                )
                axs[-3 + j].set_ylim([0.9, 50])
                axs[-3 + j].text(
                    0.97,
                    0.02,
                    wave_labs[j],
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    transform=axs[-3 + j].transAxes,
                    fontsize=10,
                    color="white",
                )
        ymin, ymax = axs[0].get_ylim()

        # plot probability
        clrs1 = ["green", "gold"]
        clrs2 = ["blue", "red"]
        if model.name == "EQTransformer":
            local_ground_true = sample["detections"][0][
                local_start_sample:local_end_sample
            ]
            assert len(local_sample) == len(local_ground_true)
            # print(
            #     len((start_sample + np.arange(len(local_ground_true))) / sampling_rate)
            # )
            # print(len((start_sample + np.arange(len(local_sample))) / sampling_rate))
            axs[1].plot(
                (np.arange(len(local_ground_true))) / sampling_rate,
                local_ground_true,
                label="Ground truth detection",
                linestyle="dashed",
                color="sandybrown",
            )
            dets = probs[idx][2]
            axs[1].plot(
                (np.arange(len(dets))) / sampling_rate,
                dets,
                label="Predicted detection",
                color="chocolate",
            )
        # elif model.name == "VolEQTransformer":
        #     rg_det_local_ground_true = sample["rg_detections"][0][
        #         local_start_sample:local_end_sample
        #     ]
        #     axs[2].plot(
        #         (np.arange(len(rg_det_local_ground_true))) / sampling_rate,
        #         rg_det_local_ground_true,
        #         label="VT ground truth",
        #         linestyle="dashed",
        #         color="sandybrown",
        #     )
        #     rg_dets = probs[idx][2]
        #     axs[2].plot(
        #         (np.arange(len(rg_dets))) / sampling_rate,
        #         rg_dets,
        #         label="VT detection",
        #         color="chocolate",
        #     )

        #     lp_det_local_ground_true = sample["lp_detections"][0][
        #         local_start_sample:local_end_sample
        #     ]
        #     axs[2].plot(
        #         (np.arange(len(lp_det_local_ground_true))) / sampling_rate,
        #         lp_det_local_ground_true,
        #         label="LP ground truth",
        #         linestyle="dashed",
        #         color="plum",
        #     )
        #     lp_dets = probs[idx][3]
        #     axs[2].plot(
        #         (np.arange(len(lp_dets))) / sampling_rate,
        #         lp_dets,
        #         label="LP detection",
        #         color="purple",
        #     )
        #     axs[2].set_ylim([0, 1.03])
        #     axs[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.0, 0.2, 1))

        for j in range(2):
            try:
                if model.name == "EQTransformer":
                    # or model.name == "VolEQTransformer":
                    lab_id = j
                else:
                    lab_id = model.labels.index(labels[j])
                # predicted probability
                prob = probs[idx][lab_id]
                axs[1].plot(
                    (np.arange(len(prob))) / sampling_rate,
                    prob,
                    label="Predicted " + labels[j],
                    color=clrs2[j],
                )
            except IndexError:
                print(f"{idx}, {j}, {lab_id}, {probs[idx].shape}")

            # ground true probability
            local_ground_true = sample["y"][j][local_start_sample:local_end_sample]
            axs[1].plot(
                (np.arange(len(local_ground_true))) / sampling_rate,
                local_ground_true,
                label="Ground truth " + labels[j],
                linestyle="dashed",
                color=clrs1[j],
            )

        # plot predicted picks
        p_ids = p_idxs_predicted[idx]
        s_ids = s_idxs_predicted[idx]

        if len(p_ids) != 0:
            # Subtracting start_sample is due to the design of SteerGenerator
            axs[0].vlines(
                x=(p_ids - start_sample) / sampling_rate,
                ymin=ymin,
                ymax=ymax,
                color="blue",
                label="Predicted P",
            )
        if len(s_ids) != 0:
            axs[0].vlines(
                x=(s_ids - start_sample) / sampling_rate,
                ymin=ymin,
                ymax=ymax,
                color="red",
                label="Predicted S",
            )

        table = [
            [tps_p[idx], fps_p[idx], fns_p[idx]],
            [tps_s[idx], fps_s[idx], fns_s[idx]],
        ]
        cell_text = [["", "TP", "FP", "FN"]]
        row_labels = [["P"], ["S"]]
        for row in range(2):
            cell_text.append(row_labels[row] + [f"{int(x)}" for x in table[row]])
        the_table = axs[0].table(
            cellText=cell_text,
            #  rowLabels=["P","S"],
            #  rowColours=["blue","red"],
            #  colLabels=["","TP","FP","FN"],
            #  colWidths=[0.25,0.25,0.25],
            cellLoc="center",
            loc="top",
            #  edges="open"
            #  bbox=[0.5,1,.6,.3]
        )
        for row in range(3):
            for col in range(4):
                the_table[row, col].set(edgecolor="dimgray")
                the_table[row, col].get_text().set_color("dimgray")
        the_table[0, 0].set(edgecolor="white")

        # plot true picks
        # get true picks (unit: sample)
        control = generator.metadata.iloc[idx].to_dict()
        kwargs = {
            "trace_name": control["trace_name"],
            "chunk": control.get("trace_chunk", None),
            "dataset": control.get("trace_dataset", None),
        }
        data_idx = generator.dataset.get_idx_from_trace_name(**kwargs)
        p_id_ground_truth = generator.dataset.get_sample(data_idx)[1][
            "trace_p_arrival_sample"
        ]
        s_id_ground_truth = generator.dataset.get_sample(data_idx)[1][
            "trace_s_arrival_sample"
        ]
        # plotting
        if not np.isnan(p_id_ground_truth):
            axs[0].vlines(
                (p_id_ground_truth - start_sample) / sampling_rate,
                ymin=ymin,
                ymax=ymax,
                linestyle="dashed",
                color=clrs1[0],
                label="Ground truth P",
            )
        if not np.isnan(s_id_ground_truth):
            axs[0].vlines(
                (s_id_ground_truth - start_sample) / sampling_rate,
                ymin=ymin,
                ymax=ymax,
                linestyle="dashed",
                color=clrs1[1],
                label="Ground truth S",
            )

        axs[0].xaxis.set_major_locator(MultipleLocator(500 / sampling_rate))
        axs[0].xaxis.set_minor_locator(MultipleLocator(50 / sampling_rate))

        # axs[1].hlines([prob_thre, prob_thre/2], start_sample, start_sample+len(probs[idx][0])-1, color=['black','darkgray'], linestyle='dashdot',label=["Higher threshold", "Lower threshold"])
        axs[1].hlines(
            prob_thre,
            0,
            (len(probs[idx][0]) - 1) / sampling_rate,
            color="black",
            linestyle="dashdot",
            label="Threshold",
        )
        # axs[1].hlines(
        #     prob_thre / 2,
        #     start_sample,
        #      len(probs[idx][0]) - 1,
        #     color="gray",
        #     linestyle="dashdot",
        #     label="Lower threshold",
        # )

        axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 0.1, 0.2, 1))
        axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.0, 0.2, 1))

        # axs[3].legend(loc="upper left", bbox_to_anchor=(1.0, 0.0, 0.2, 1))

        # axs[0].set_title("Normalized waveform")
        # axs[1].set_title("Probability")

        xmin = np.min(np.arange(len(local_sample))) / sampling_rate
        xmax = np.max(np.arange(len(local_sample))) / sampling_rate
        axs[0].set_xlim([xmin, xmax])
        axs[1].set_ylim([0, 1.03])
        # legend(loc="upper right", bbox_to_anchor=(-0.14, 1.05, 1, 0.18), ncol=3,columnspacing=0.8,fontsize=fts-1.5, frameon=False)
        # plot thresholds

        # axs[-1]

        # print(p_ids)
        # print(p_id_ground_truth)
        # print(len(sample["X"][j][start_sample:end_sample]))
        trace_idx = generator.metadata.iloc[idx]["trace_idx"]
        if not fig_dir.exists():
            fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / f"waveform_{trace_idx}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
