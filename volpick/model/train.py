"""
This script handles the training of models base on model configuration files.
"""

import seisbench.generate as sbg
from seisbench.util import worker_seeding

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# https://github.com/Lightning-AI/lightning/pull/12554
# https://github.com/Lightning-AI/lightning/issues/11796
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
    EarlyStopping,
    LearningRateFinder,
)
from pytorch_lightning.tuner import Tuner


import packaging
import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import logging

import volpick.data.utils as data_utils
import models
import seisbench.models as sbm
import time
import datetime
from pathlib import Path
from volpick.model.ema import EMA, EMAModelCheckpoint


# class FineTuneLearningRateFinder(LearningRateFinder):
#     def __init__(self, milestones, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.milestones = milestones

#     def on_fit_start(self, *args, **kwargs):
#         return

#     def on_train_epoch_start(self, trainer, pl_module):
#         if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
#             self.lr_find(trainer, pl_module)
#             # Plot with
#             fig = self.optimal_lr.plot(suggest=True)
#             fig.savefig(
#                 Path(trainer.logger.log_dir) / f"lr_finder{trainer.current_epoch}.jpg",
#                 dpi=300,
#             )
#             print(self.optimal_lr.suggestion())
#             print(pl_module.lr)


def train(config, experiment_name, test_run):
    """
    Runs the model training defined by the config.

    Config parameters:

        - model: Model used as in the models.py file, but without the Lit suffix
        - data: Dataset used, as in seisbench.data
        - model_args: Arguments passed to the constructor of the model lightning module
        - trainer_args: Arguments passed to the lightning trainer
        - batch_size: Batch size for training and validation
        - num_workers: Number of workers for data loading, default to 12.
        - restrict_to_phase: Filters datasets only to examples containing the given phase.
          By default, uses all phases.
        - training_fraction: Fraction of training blocks to use as float between 0 and 1. Defaults to 1.

    :param config: Configuration parameters for training
    :param test_run: If true, makes a test run with less data and less logging. Intended for debug purposes.
    """
    code_start_time = time.perf_counter()

    monitored_loss = "val_loss"
    if config["whole_dataset"]:
        monitored_loss = "train_loss"
        print("Training on the whole dataset: monitoring training loss only.")

    model = models.__getattribute__(config["model"] + "Lit")(
        lr_monitor=monitored_loss, **config.get("model_args", {})
    )
    pretrained = config.get("pretrained", None)
    if pretrained is not None:
        model.model = sbm.__getattribute__(config["model"]).from_pretrained(pretrained)
        print(
            f"""Loading {config["model"]} pretrained on {pretrained} to initialized the weights"""
        )
        print(
            f"The normalization type will be superseded by the one for the pretrained model. The normalization type specified by the configuration file will be ignored"
        )
        print(f"Nomalization type: {model.model.norm}")
    else:
        print("No pretrained model is used. The model will be trained from scratch.")
    print()

    print(f"""Learning rate: {config["model_args"].get("lr",0.01)}""")
    print(
        f"""Labelling function: {config["model_args"].get("prob_label_shape","gaussian")}"""
    )
    print(f"""Labeling width (sigma): {config["model_args"].get("sigma",20)}""")
    print(f"""Max epochs: {config["trainer_args"]["max_epochs"]}""")
    print(
        f"""Using the augmentation of superimposing events/noise: {config.get("stack_data", False)}"""
    )
    train_loader, dev_loader = prepare_data(config, model, test_run)
    print(f"{len(train_loader)} batches in the training data loader.")

    # CSV logger - also used for saving configuration as yaml
    save_dir = config.get("save_dir", "weights")
    csv_logger = CSVLogger(save_dir, experiment_name)
    csv_logger.log_hyperparams(config)
    loggers = [csv_logger]

    default_root_dir = os.path.join(
        save_dir
    )  # Experiment name is parsed from the loggers
    if not test_run:
        tb_logger = TensorBoardLogger("tb_logs", experiment_name)
        tb_logger.log_hyperparams(config)
        loggers += [tb_logger]

    if config.get("whole_dataset", None):
        checkpoint_every_n_train_steps = 5
    else:
        checkpoint_every_n_train_steps = None

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        filename="{epoch}-{step}",
        monitor=monitored_loss,
        mode="min",
        every_n_train_steps=checkpoint_every_n_train_steps,
    )  # save_top_k=1, monitor="val_loss", mode="min": save the best model in terms of validation loss
    lr_monitor = LearningRateMonitor(
        logging_interval="step"
    )  # Monitoring learning rate
    callbacks = [checkpoint_callback, lr_monitor]

    if config.get("swa", None):
        print("Enable stochastic weight averaging")
        callbacks.append(StochasticWeightAveraging(**config.get("swa", {})))
    if config.get("ema", False):
        emadecay = 0.999
        print(f"Enable EMA. Decay: {emadecay}.")
        emacallback = EMA(
            decay=emadecay,
            validate_original_weights=False,
            every_n_steps=1,
            cpu_offload=False,
        )
        # emacallback = EMA(
        #     decay=emadecay,
        #     apply_ema_every_n_steps=1,
        #     start_step=0,
        #     save_ema_weights_in_callback_state=True,
        #     evaluate_ema_weights_instead=True,
        # )
        checkpoint_callback = EMAModelCheckpoint(
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{step}",
            monitor=monitored_loss,
            mode="min",
        )
        callbacks = [checkpoint_callback, lr_monitor, emacallback]
    if config.get("early_stop", False):
        callbacks.append(
            EarlyStopping(monitor=monitored_loss, patience=100, mode="min")
        )

    ## Uncomment the following 2 lines to enable
    # device_stats = DeviceStatsMonitor()
    # callbacks.append(device_stats)

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=5,
        **config.get("trainer_args", {}),
    )

    if config.get("auto_lr", False):
        # Create a Tuner
        tuner = Tuner(trainer)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        lr_finder = tuner.lr_find(
            model, train_loader, dev_loader, min_lr=1e-5, max_lr=1e-2, num_training=200
        )
        fig = lr_finder.plot(suggest=True)
        fig.savefig(Path(trainer.logger.log_dir) / "lr_finder.jpg", dpi=300)
        print(model.lr)

    trainer.fit(model, train_loader, dev_loader)

    running_time = str(
        datetime.timedelta(seconds=time.perf_counter() - code_start_time)
    )
    print(f"Running time: {running_time}")
    with open(
        Path(trainer.logger.log_dir) / "running_time.txt", "w"
    ) as running_time_file:
        running_time_file.write(f"{running_time}")

    if config.get("training_fraction", 1.0) < 1:
        with open(
            Path(trainer.logger.log_dir) / "num_training_examples.txt", "w"
        ) as num_training_file:
            num_training_file.write(f"{len(train_loader.dataset)}")


def prepare_data(config, model, test_run):
    """
    Returns the training and validation data loaders
    :param config:
    :param model:
    :param test_run:
    :return:
    """
    batch_size = config.get("batch_size", 1024)
    num_workers = config.get("num_workers", 12)
    print(f"Batch size: {batch_size}")
    print(f"Number of workers for data loading: {num_workers}")
    if config["read_data_method"] == "name":
        dataset = data_utils.get_dataset_by_name(config["data"])(
            sampling_rate=100,
            component_order="ZNE",
            dimension_order="NCW",
            cache="full",
        )
    else:
        dataset = data_utils.get_dataset_by_path(config["data"])
    restrict_to_phase = config.get("restrict_to_phase", None)
    if restrict_to_phase is not None:
        mask = generate_phase_mask(dataset, restrict_to_phase)
        dataset.filter(mask, inplace=True)

    remove_spikes = config.get("remove_spikes", False)
    if remove_spikes:
        mask = ~dataset.metadata["trace_has_spikes"]
        dataset.filter(mask, inplace=True)

    if "split" not in dataset.metadata.columns:
        logging.warning("No split defined, adding auxiliary split.")
        split = np.array(["train"] * len(dataset))
        split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
        split[int(0.7 * len(dataset)) :] = "test"

        dataset._metadata["split"] = split

    if config["whole_dataset"]:
        print(
            "Train the model on the whole data set. "
            "The model can be trained on the whole data set only after"
            "we have trained and tested a model using a train/validation/test split."
        )
        split = np.array(["train"] * len(dataset))
        dataset._metadata["split"] = split

    train_data = dataset.train()
    dev_data = dataset.dev()

    if test_run:
        # Only use a small part of the dataset
        train_mask = np.zeros(len(train_data), dtype=bool)
        train_mask[:1000] = True
        train_data.filter(train_mask, inplace=True)

        dev_mask = np.zeros(len(dev_data), dtype=bool)
        dev_mask[:1000] = True
        dev_data.filter(dev_mask, inplace=True)
        batch_size = config["batch_size"] = 10

    training_fraction = config.get("training_fraction", 1.0)
    apply_training_fraction(training_fraction, train_data)

    train_data.preload_waveforms(pbar=True)
    dev_data.preload_waveforms(pbar=True)

    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)

    if config.get("stack_data", False) == True:
        # print("Using the augmentation block of stacking noise and earthquake waveforms")
        train_data_eq = train_data.filter(
            train_data["source_type"] != "noise", inplace=False
        )
        train_data_noise = train_data.filter(
            train_data["source_type"] == "noise", inplace=False
        )
        val_data_eq = dev_data.filter(dev_data["source_type"] != "noise", inplace=False)
        val_data_noise = dev_data.filter(
            dev_data["source_type"] == "noise", inplace=False
        )
        model.set_superimposed_event_generator(
            train_data_eq, train_data_noise, val_data_eq, val_data_noise
        )

    train_generator.add_augmentations(model.get_train_augmentations())
    dev_generator.add_augmentations(model.get_val_augmentations())

    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=worker_seeding,
        drop_last=True,  # Avoid crashes from batch norm layers for batch size 1
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=worker_seeding,
    )

    return train_loader, dev_loader


def apply_training_fraction(training_fraction, train_data):
    """
    Reduces the size of train_data to train_fraction by inplace filtering.
    Filter blockwise for efficient memory savings.

    :param training_fraction: Training fraction between 0 and 1.
    :param train_data: Training dataset
    :return: None
    """

    if not 0.0 < training_fraction <= 1.0:
        raise ValueError("Training fraction needs to be between 0 and 1.")

    if training_fraction < 1:
        print(
            f"Reduces the size of train_data to {training_fraction} by inplace filtering."
        )
        blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
        unique_blocks = blocks.unique()
        np.random.seed(42)
        np.random.shuffle(unique_blocks)
        target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
        target_blocks = set(target_blocks)
        mask = blocks.isin(target_blocks)
        train_data.filter(mask, inplace=True)


def generate_phase_mask(dataset, phases):
    mask = np.zeros(len(dataset), dtype=bool)

    for key, phase in models.phase_dict.items():
        if phase not in phases:
            continue
        else:
            if key in dataset.metadata:
                mask = np.logical_or(mask, ~np.isnan(dataset.metadata[key]))

    return mask


if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="If true, makes a test run with less data and less logging. "
        "Intended for debug purposes.",
    )
    parser.add_argument(
        "--whole_dataset",
        action="store_true",
        help="If true, train the model on the whole data set. "
        "The model can be trained on the whole data set only after "
        "we have trained and tested a model using a train/validation/test split.",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Learning rate. If passed from the command line, "
        "it will override the learning rate in the configuration file",
    )
    parser.add_argument(
        "--fraction",
        default=None,
        type=float,
        help="Fraction of training blocks to use as float between 0 and 1. "
        "Defaults to 1. If passed from the command line,  it will override "
        "the learning rate in the configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    experiment_name = os.path.basename(args.config)[:-5]
    if args.fraction is not None:
        logging.warning(f"Overwriting training fraction to {args.fraction}")
        experiment_name += f"_frac{args.fraction:.3f}"
        config["training_fraction"] = args.fraction
    if args.lr is not None:
        logging.warning(f"Overwriting learning rate to {args.lr}")
        experiment_name += f"_{args.lr}"
        config["model_args"]["lr"] = args.lr
    if not "read_data_method" in config:
        config["read_data_method"] = "name"
    if args.test_run:
        experiment_name = experiment_name + "_test"
    config["whole_dataset"] = args.whole_dataset

    train(config, experiment_name, test_run=args.test_run)
