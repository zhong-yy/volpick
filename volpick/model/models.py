"""
This file contains pytorch-lightning modules for PhaseNet and EQTransformer 

This file is modified from https://github.com/seisbench/pick-benchmark/blob/main/benchmark/models.py
"""

import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod, ABC

import seisbench
from seisbench.generate.labeling import SupervisedLabeller

# Allows to import this file in both jupyter notebook and code
try:
    from .augmentations import MyDuplicateEvent, SuperimposeEvent
except ImportError:
    from augmentations import MyDuplicateEvent, SuperimposeEvent

# Phase dict for labelling
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
}


def vector_cross_entropy(y_pred, y_true, eps=1e-5):
    """
    Cross entropy loss

    :param y_true: True label probabilities
    :param y_pred: Predicted label probabilities
    :param eps: Epsilon to clip values for stability
    :return: Average loss across batch
    """
    h = y_true * torch.log(y_pred + eps)
    if y_pred.ndim == 3:
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
    else:
        h = h.sum(-1)  # Sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


class SeisBenchModuleLit(pl.LightningModule, ABC):
    """
    Abstract interface for SeisBench lightning modules.
    Adds generic function, e.g., get_augmentations
    """

    @abstractmethod
    def get_augmentations(self):
        """
        Returns a list of augmentations that can be passed to the seisbench.generate.GenericGenerator

        :return: List of augmentations
        """
        pass

    def get_train_augmentations(self):
        """
        Returns the set of training augmentations.
        """
        return self.get_augmentations()

    def get_val_augmentations(self):
        """
        Returns the set of validation augmentations for validations during training.
        """
        return self.get_augmentations()

    @abstractmethod
    def get_eval_augmentations(self):
        """
        Returns the set of evaluation augmentations for evaluation after training.
        These augmentations will be passed to a SteeredGenerator and should usually contain a steered window.
        """
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        """
        Predict step for the lightning module. Returns results for three tasks:

        - earthquake detection (score, higher means more likely detection)
        - P to S phase discrimination (score, high means P, low means S)
        - phase location in samples (two integers, first for P, second for S wave)

        All predictions should only take the window defined by batch["window_borders"] into account.

        :param batch:
        :return:
        """
        score_detection = None
        score_p_or_s = None
        p_sample = None
        s_sample = None
        return score_detection, score_p_or_s, p_sample, s_sample


class PhaseNetLit(SeisBenchModuleLit):
    """
    LightningModule for PhaseNet

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param kwargs: Kwargs are passed to the SeisBench.models.PhaseNet constructor.
    :param prob_label_shape: "gaussian"|"triangle"|"box"
    """

    def __init__(
        self,
        lr=1e-2,
        sigma=20,
        prob_label_shape="gaussian",
        sample_boundaries=(None, None),
        rotate_array=False,
        lr_scheduler=None,
        lr_scheduler_args=None,
        lr_monitor="val_loss",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.prob_label_shape = prob_label_shape
        self.sigma = sigma

        self.sample_boundaries = sample_boundaries
        self.loss = vector_cross_entropy
        self.model = sbm.PhaseNet(**kwargs)
        self.rotate_array = rotate_array
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.lr_monitor = lr_monitor

        self.train_stacked_event_generator = None
        self.train_stacked_noise_generator = None
        self.val_stacked_event_generator = None
        self.val_stacked_noise_generator = None

        # For the same sigma, the width of the triangle function is about a half of
        # that of the gaussian function
        if prob_label_shape == "triangle":
            self.sigma = 2 * self.sigma

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"]
        y_pred = self.model(x)
        return self.loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler:
            learning_rate_scheduler = torch.optim.lr_scheduler.__getattribute__(
                self.lr_scheduler
            )(optimizer=optimizer, **self.lr_scheduler_args)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    # REQUIRED: The scheduler instance
                    "scheduler": learning_rate_scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": self.lr_monitor,
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": False,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    "name": None,
                },
            }
        else:
            return optimizer

    def get_joint_augmentation_block1(
        self,
        first_window_kwargs={
            "samples_before": 3000,
            "windowlen": 6000,
            "selection": "random",
        },
        first_window_prob=[2, 1],
    ):
        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=first_window_kwargs["samples_before"],
                        windowlen=first_window_kwargs["windowlen"],
                        selection=first_window_kwargs["selection"],
                        # strategy="variable",
                        strategy="pad",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=first_window_prob,
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=3001,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(
                shape=self.prob_label_shape,
                label_columns=phase_dict,
                noise_column=True,
                sigma=self.sigma,
                dim=0,
            ),
            sbg.Normalize(
                demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
            ),  # "peak"/"std"
        ]
        return block1

    def get_joint_augmentation_block2(self):
        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
        ]
        return block2

    def set_superimposed_event_generator(
        self, train_data_eq, train_data_noise, val_data_eq, val_data_noise
    ):
        block1_aug = self.get_joint_augmentation_block1(
            {"samples_before": 1500, "windowlen": 4000, "selection": "first"}, [1, 0]
        )
        block2_aug = self.get_joint_augmentation_block1()
        if len(train_data_eq) > 0:
            if not np.all(train_data_eq["split"] == "train"):
                raise KeyError("'train_data_eq' should be a subset of the training set")
            self.train_stacked_event_generator = sbg.GenericGenerator(train_data_eq)
            self.train_stacked_event_generator.add_augmentations(block1_aug)
        else:
            print("There are no earthquake waveforms in the training set.")

        if len(train_data_noise) > 0:
            if not np.all(train_data_noise["split"] == "train"):
                raise KeyError("'train_data_eq' should be a subset of the training set")
            self.train_stacked_noise_generator = sbg.GenericGenerator(train_data_noise)
            self.train_stacked_noise_generator.add_augmentations(block2_aug)
        # else:
        #     print("There are no noise waveforms in the training set.")

        if len(val_data_eq) > 0:
            if not np.all(val_data_eq["split"] == "dev"):
                raise KeyError(
                    "'val_data_eq' should be a subset of the development set"
                )
            self.val_stacked_event_generator = sbg.GenericGenerator(val_data_eq)
            self.val_stacked_event_generator.add_augmentations(block1_aug)
        else:
            print("There are no earthquake waveforms in the validation set.")

        if len(val_data_noise) > 0:
            if not np.all(val_data_noise["split"] == "dev"):
                raise KeyError(
                    "'val_data_noise' should be a subset of the development set"
                )
            self.val_stacked_noise_generator = sbg.GenericGenerator(val_data_noise)
            self.val_stacked_noise_generator.add_augmentations(block2_aug)
        # else:
        #     print("There are no noise waveforms in the validation set.")

    def get_gap_augmentation_block(self):
        # Gaps
        gap_augmentation_block = [
            sbg.OneOf(
                [
                    sbg.AddGap(label_keys="y"),
                    sbg.NullAugmentation(),
                ],
                [0.2, 0.8],
            )
        ]
        return gap_augmentation_block

    def get_rotation_block(self):
        if self.rotate_array:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []
        return rotation_block

    def get_stack_block(self, event_generator, noise_generator):
        stack_event_augmentation_block = []
        if event_generator is not None:
            stack_event_augmentation_block.append(
                sbg.OneOf(
                    [
                        SuperimposeEvent(
                            data_generator=event_generator,
                            label_columns=phase_dict,
                            inv_scale=(0.25, 4),
                            key="X",
                            label_key="y",
                            detection_keys=None,
                            sep=200,
                            noise_label=True,
                            noise_data=False,
                        ),
                        MyDuplicateEvent(
                            label_columns=phase_dict,
                            inv_scale=(0.25, 4),
                            key="X",
                            label_key="y",
                            detection_keys=None,
                            sep=200,
                            noise_label=True,
                        ),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[0.2, 0.2, 0.6],
                )
            )
        if noise_generator is not None:
            stack_event_augmentation_block.append(
                sbg.OneOf(
                    [
                        SuperimposeEvent(
                            data_generator=noise_generator,
                            label_columns=phase_dict,
                            key="X",
                            label_key="y",
                            detection_keys=None,
                            inv_scale=(2, 50),
                            sep=200,
                            noise_label=True,
                            noise_data=True,
                        ),
                        sbg.GaussianNoise(),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[0.25, 0.25, 0.5],
                )
            )
        return stack_event_augmentation_block

    def get_train_augmentations(self):
        stack_event_augmentation_block = self.get_stack_block(
            self.train_stacked_event_generator, self.train_stacked_noise_generator
        )
        block1 = self.get_joint_augmentation_block1()
        block2 = self.get_joint_augmentation_block2()

        gap_augmentation_block = self.get_gap_augmentation_block()
        rotation_block = self.get_rotation_block()
        return (
            block1
            + stack_event_augmentation_block
            + rotation_block
            + gap_augmentation_block
            + [  # Augmentations make second normalize necessary
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
                )
            ]
            + block2
        )

    def get_val_augmentations(self):
        block1 = self.get_joint_augmentation_block1()
        block2 = self.get_joint_augmentation_block2()
        stack_event_augmentation_block = self.get_stack_block(
            self.val_stacked_event_generator, self.val_stacked_noise_generator
        )
        gap_augmentation_block = self.get_gap_augmentation_block()
        rotation_block = self.get_rotation_block()
        return (
            block1
            + stack_event_augmentation_block
            + rotation_block
            + gap_augmentation_block
            + [  # Augmentations make second normalize necessary
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
                )
            ]
            + block2
        )

    def get_augmentations(self):
        raise NotImplementedError("Use get_train/val_augmentations instead.")

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=3001, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(
                demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
            ),
        ]

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        pred = self.model(x)

        score_detection = torch.zeros(pred.shape[0])
        score_p_or_s = torch.zeros(pred.shape[0])
        p_sample = torch.zeros(pred.shape[0], dtype=int)
        s_sample = torch.zeros(pred.shape[0], dtype=int)
        p_label_id = self.model.labels.index("P")
        s_label_id = self.model.labels.index("S")
        noise_label_id = self.model.labels.index("N")

        for i in range(pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, :, start_sample:end_sample]

            score_detection[i] = torch.max(1 - local_pred[noise_label_id])  # 1 - noise
            score_p_or_s[i] = torch.max(local_pred[p_label_id]) / torch.max(
                local_pred[s_label_id]
            )  # most likely P by most likely S

            p_sample[i] = torch.argmax(local_pred[p_label_id])
            s_sample[i] = torch.argmax(local_pred[s_label_id])

        return score_detection, score_p_or_s, p_sample, s_sample


class EQTransformerLit(SeisBenchModuleLit):
    """
    LightningModule for EQTransformer

    :param lr: Learning rate, defaults to 1e-2
    :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
    :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
    :param loss_weights: Loss weights for detection, P and S phase.
    :param rotate_array: If true, rotate array along sample axis.
    :param detection_fixed_window: Passed as parameter fixed_window to detection
    :param kwargs: Kwargs are passed to the SeisBench.models.EQTransformer constructor.
    """

    def __init__(
        self,
        lr=1e-2,
        sigma=20,
        sample_boundaries=(None, None),
        loss_weights=(0.05, 0.40, 0.55),
        rotate_array=False,
        detection_fixed_window=None,
        prob_label_shape="gaussian",
        lr_scheduler=None,
        lr_scheduler_args=None,
        lr_monitor="val_loss",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.prob_label_shape = prob_label_shape
        self.sigma = sigma
        self.sample_boundaries = sample_boundaries
        self.loss = torch.nn.BCELoss()
        self.loss_weights = loss_weights
        self.rotate_array = rotate_array
        self.detection_fixed_window = detection_fixed_window
        self.model = sbm.EQTransformer(**kwargs)
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.lr_monitor = lr_monitor

        self.train_stacked_event_generator = None
        self.train_stacked_noise_generator = None
        self.val_stacked_event_generator = None
        self.val_stacked_noise_generator = None

        # For the same sigma, the width of the triangle function is about a half of
        # that of the gaussian function
        if prob_label_shape == "triangle":
            self.sigma = 2 * self.sigma

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x = batch["X"]
        p_true = batch["y"][:, 0]
        s_true = batch["y"][:, 1]
        det_true = batch["detections"][:, 0]
        det_pred, p_pred, s_pred = self.model(x)

        return (
            self.loss_weights[0] * self.loss(det_pred, det_true)
            + self.loss_weights[1] * self.loss(p_pred, p_true)
            + self.loss_weights[2] * self.loss(s_pred, s_true)
        )

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_scheduler:
            learning_rate_scheduler = torch.optim.lr_scheduler.__getattribute__(
                self.lr_scheduler
            )(optimizer=optimizer, **self.lr_scheduler_args)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    # REQUIRED: The scheduler instance
                    "scheduler": learning_rate_scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": self.lr_monitor,
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": True,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    "name": None,
                },
            }
        else:
            return optimizer

    def get_joint_augmentation_block1(
        self,
        first_window_kwargs={
            "samples_before": 6000,
            "windowlen": 12000,
            "selection": "random",
        },
        first_window_prob=[2, 1],
    ):
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]

        if self.detection_fixed_window is not None:
            detection_labeller = sbg.DetectionLabeller(
                p_phases,
                fixed_window=self.detection_fixed_window,
                key=("X", "detections"),
            )
        else:
            detection_labeller = sbg.DetectionLabeller(
                p_phases, s_phases=s_phases, key=("X", "detections")
            )

        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=first_window_kwargs["samples_before"],
                        windowlen=first_window_kwargs["windowlen"],
                        selection=first_window_kwargs["selection"],
                        # strategy="variable",
                        strategy="pad",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=first_window_prob,
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=6000,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(
                shape=self.prob_label_shape,
                label_columns=phase_dict,
                noise_column=False,
                sigma=self.sigma,
                dim=0,
            ),
            detection_labeller,
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(
                detrend_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
            ),  # "peak"/“std
        ]
        return block1

    def get_joint_augmentation_block2(self):
        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "detections"),
        ]
        return block2

    def set_superimposed_event_generator(
        self, train_data_eq, train_data_noise, val_data_eq, val_data_noise
    ):
        block1_aug = self.get_joint_augmentation_block1(
            {"samples_before": 3000, "windowlen": 8000, "selection": "first"}, [1, 0]
        )
        block2_aug = self.get_joint_augmentation_block1()
        if len(train_data_eq) > 0:
            if not np.all(train_data_eq["split"] == "train"):
                raise KeyError("'train_data_eq' should be a subset of the training set")
            self.train_stacked_event_generator = sbg.GenericGenerator(train_data_eq)
            self.train_stacked_event_generator.add_augmentations(block1_aug)
        else:
            print("There are no earthquake waveforms in the training set.")

        if len(train_data_noise) > 0:
            if not np.all(train_data_noise["split"] == "train"):
                raise KeyError("'train_data_eq' should be a subset of the training set")
            self.train_stacked_noise_generator = sbg.GenericGenerator(train_data_noise)
            self.train_stacked_noise_generator.add_augmentations(block2_aug)
        # else:
        #     print("There are no noise waveforms in the training set.")

        if len(val_data_eq) > 0:
            if not np.all(val_data_eq["split"] == "dev"):
                raise KeyError(
                    "'val_data_eq' should be a subset of the development set"
                )
            self.val_stacked_event_generator = sbg.GenericGenerator(val_data_eq)
            self.val_stacked_event_generator.add_augmentations(block1_aug)
        else:
            print("There are no earthquake waveforms in the validation set.")

        if len(val_data_noise) > 0:
            if not np.all(val_data_noise["split"] == "dev"):
                raise KeyError(
                    "'val_data_noise' should be a subset of the development set"
                )
            self.val_stacked_noise_generator = sbg.GenericGenerator(val_data_noise)
            self.val_stacked_noise_generator.add_augmentations(block2_aug)
        # else:
        #     print("There are no noise waveforms in the validation set.")

    def get_gap_augmentation_block(self):
        # Gaps
        gap_augmentation_block = [
            sbg.OneOf(
                [
                    sbg.AddGap(label_keys=["y", "detections"], noise_id={}),
                    sbg.NullAugmentation(),
                ],
                [0.2, 0.8],
            )
        ]
        return gap_augmentation_block

    def get_rotation_block(self):
        if self.rotate_array:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []
        return rotation_block

    def get_stack_block(self, event_generator, noise_generator):
        stack_event_augmentation_block = []
        if event_generator is not None:
            stack_event_augmentation_block.append(
                sbg.OneOf(
                    [
                        SuperimposeEvent(
                            data_generator=event_generator,
                            label_columns=phase_dict,
                            inv_scale=(0.25, 4),
                            key="X",
                            label_key="y",
                            detection_keys=["detections"],
                            sep=200,
                            noise_label=False,
                            noise_data=False,
                        ),
                        MyDuplicateEvent(
                            label_columns=phase_dict,
                            inv_scale=(0.25, 4),
                            key="X",
                            label_key="y",
                            detection_keys=["detections"],
                            sep=200,
                            noise_label=False,
                        ),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[0.2, 0.2, 0.6],
                )
            )
        if noise_generator is not None:
            stack_event_augmentation_block.append(
                sbg.OneOf(
                    [
                        SuperimposeEvent(
                            data_generator=noise_generator,
                            label_columns=phase_dict,
                            key="X",
                            label_key="y",
                            detection_keys=None,
                            inv_scale=(2, 50),
                            sep=200,
                            noise_label=False,
                            noise_data=True,
                        ),
                        sbg.GaussianNoise(),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[0.25, 0.25, 0.5],
                )
            )

        return stack_event_augmentation_block

    def get_train_augmentations(self):
        stack_event_augmentation_block = self.get_stack_block(
            self.train_stacked_event_generator, self.train_stacked_noise_generator
        )
        block1 = self.get_joint_augmentation_block1()
        block2 = self.get_joint_augmentation_block2()

        gap_augmentation_block = self.get_gap_augmentation_block()
        rotation_block = self.get_rotation_block()
        return (
            block1
            + stack_event_augmentation_block
            + rotation_block
            + gap_augmentation_block
            + [  # Augmentations make second normalize necessary
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
                )
            ]
            + block2
        )

    def get_val_augmentations(self):
        stack_event_augmentation_block = self.get_stack_block(
            self.val_stacked_event_generator, self.val_stacked_noise_generator
        )
        block1 = self.get_joint_augmentation_block1()
        block2 = self.get_joint_augmentation_block2()

        gap_augmentation_block = self.get_gap_augmentation_block()
        rotation_block = self.get_rotation_block()
        return (
            block1
            + stack_event_augmentation_block
            + rotation_block
            + gap_augmentation_block
            + [  # Augmentations make second normalize necessary
                sbg.Normalize(
                    demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
                )
            ]
            + block2
        )

    def get_augmentations(self):
        raise NotImplementedError("Use get_train/val_augmentations instead.")

    def get_eval_augmentations(self):
        return [
            sbg.SteeredWindow(windowlen=6000, strategy="pad"),
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(
                demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
            ),
        ]
        # block1 = self.get_joint_augmentation_block1()
        # block2 = self.get_joint_augmentation_block2()
        # if (
        #     self.val_stacked_event_generator is not None
        #     and self.val_stacked_noise_generator is not None
        # ):
        #     stack_event_augmentation_block = self.get_stack_block(
        #         self.val_stacked_event_generator, self.val_stacked_noise_generator
        #     )
        # else:
        #     stack_event_augmentation_block = []
        # gap_augmentation_block = self.get_gap_augmentation_block()
        # return (
        #     block1
        #     + stack_event_augmentation_block
        #     + gap_augmentation_block
        #     + [  # Augmentations make second normalize necessary
        #         sbg.Normalize(
        #             demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
        #         )
        #     ]
        #     + block2
        # )

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch["X"]
        window_borders = batch["window_borders"]

        det_pred, p_pred, s_pred = self.model(x)

        score_detection = torch.zeros(det_pred.shape[0])
        score_p_or_s = torch.zeros(det_pred.shape[0])
        p_sample = torch.zeros(det_pred.shape[0], dtype=int)
        s_sample = torch.zeros(det_pred.shape[0], dtype=int)

        for i in range(det_pred.shape[0]):
            start_sample, end_sample = window_borders[i]
            local_det_pred = det_pred[i, start_sample:end_sample]
            local_p_pred = p_pred[i, start_sample:end_sample]
            local_s_pred = s_pred[i, start_sample:end_sample]

            score_detection[i] = torch.max(local_det_pred)
            score_p_or_s[i] = torch.max(local_p_pred) / torch.max(
                local_s_pred
            )  # most likely P by most likely S

            p_sample[i] = torch.argmax(local_p_pred)
            s_sample[i] = torch.argmax(local_s_pred)

        return score_detection, score_p_or_s, p_sample, s_sample


# class VolEQTransformerLit(SeisBenchModuleLit):
#     """
#     LightningModule for EQTransformer

#     :param lr: Learning rate, defaults to 1e-2
#     :param sigma: Standard deviation passed to the ProbabilisticPickLabeller
#     :param sample_boundaries: Low and high boundaries for the RandomWindow selection.
#     :param loss_weights: Loss weights for detection, P and S phase.
#     :param rotate_array: If true, rotate array along sample axis.
#     :param detection_fixed_window: Passed as parameter fixed_window to detection
#     :param kwargs: Kwargs are passed to the SeisBench.models.EQTransformer constructor.
#     """

#     def __init__(
#         self,
#         lr=1e-2,
#         sigma=20,
#         sample_boundaries=(None, None),
#         loss_weights=(0.05, 0.05, 0.45, 0.45),
#         rotate_array=False,
#         detection_fixed_window=None,
#         prob_label_shape="gaussian",
#         lr_scheduler=None,
#         lr_scheduler_args=None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.prob_label_shape = prob_label_shape
#         self.sigma = sigma
#         self.sample_boundaries = sample_boundaries
#         self.loss = torch.nn.BCELoss()
#         self.loss_weights = loss_weights
#         self.rotate_array = rotate_array
#         self.detection_fixed_window = detection_fixed_window
#         self.model = VolEQTransformer(**kwargs)
#         self.lr_scheduler = lr_scheduler
#         self.lr_scheduler_args = lr_scheduler_args

#         self.train_stacked_event_generator = None
#         self.train_stacked_noise_generator = None
#         self.val_stacked_event_generator = None
#         self.val_stacked_noise_generator = None

#         # For the same sigma, the width of the triangle function is about a half of
#         # that of the gaussian function
#         if prob_label_shape == "triangle":
#             self.sigma = 2 * self.sigma

#     def forward(self, x):
#         return self.model(x)

#     def shared_step(self, batch):
#         x = batch["X"]
#         p_true = batch["y"][:, 0]
#         s_true = batch["y"][:, 1]
#         rg_det_true = batch["rg_detections"][:, 0]
#         lp_det_true = batch["lp_detections"][:, 0]
#         rg_det_pred, lp_det_pred, p_pred, s_pred = self.model(x)

#         return (
#             self.loss_weights[0] * self.loss(rg_det_pred, rg_det_true)
#             + self.loss_weights[0] * self.loss(lp_det_pred, lp_det_true)
#             + self.loss_weights[1] * self.loss(p_pred, p_true)
#             + self.loss_weights[2] * self.loss(s_pred, s_true)
#         )

#     def training_step(self, batch, batch_idx):
#         loss = self.shared_step(batch)
#         self.log("train_loss", loss, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.shared_step(batch)
#         self.log("val_loss", loss, sync_dist=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         if self.lr_scheduler:
#             learning_rate_scheduler = torch.optim.lr_scheduler.__getattribute__(
#                 self.lr_scheduler
#             )(optimizer=optimizer, **self.lr_scheduler_args)
#             return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     # REQUIRED: The scheduler instance
#                     "scheduler": learning_rate_scheduler,
#                     # The unit of the scheduler's step size, could also be 'step'.
#                     # 'epoch' updates the scheduler on epoch end whereas 'step'
#                     # updates it after a optimizer update.
#                     "interval": "epoch",
#                     # How many epochs/steps should pass between calls to
#                     # `scheduler.step()`. 1 corresponds to updating the learning
#                     # rate after every epoch/step.
#                     "frequency": 1,
#                     # Metric to to monitor for schedulers like `ReduceLROnPlateau`
#                     "monitor": "val_loss",
#                     # If set to `True`, will enforce that the value specified 'monitor'
#                     # is available when the scheduler is updated, thus stopping
#                     # training if not found. If set to `False`, it will only produce a warning
#                     "strict": True,
#                     # If using the `LearningRateMonitor` callback to monitor the
#                     # learning rate progress, this keyword can be used to specify
#                     # a custom logged name
#                     "name": None,
#                 },
#             }
#         else:
#             return optimizer

#     def get_joint_augmentation_block1(
#         self,
#         first_window_kwargs={
#             "samples_before": 6000,
#             "windowlen": 12000,
#             "selection": "random",
#         },
#         first_window_prob=[2, 1],
#     ):
#         p_phases = [key for key, val in phase_dict.items() if val == "P"]
#         s_phases = [key for key, val in phase_dict.items() if val == "S"]

#         if self.detection_fixed_window is not None:
#             # detection_labeller = sbg.DetectionLabeller(
#             #     p_phases,
#             #     fixed_window=self.detection_fixed_window,
#             #     key=("X", "detections"),
#             # )
#             lp_detection_labeller = EventTypeDetectionLabeller(
#                 source_type="lp",
#                 p_phases=p_phases,
#                 fixed_window=self.detection_fixed_window,
#                 key=("X", "lp_detections"),
#                 sigma=self.sigma,
#             )
#             rg_detection_labeller = EventTypeDetectionLabeller(
#                 source_type="regular",
#                 p_phases=p_phases,
#                 fixed_window=self.detection_fixed_window,
#                 key=("X", "rg_detections"),
#                 sigma=self.sigma,
#             )
#         else:
#             # detection_labeller = sbg.DetectionLabeller(
#             #     p_phases, s_phases=s_phases, key=("X", "detections")
#             # )
#             lp_detection_labeller = EventTypeDetectionLabeller(
#                 source_type="lp",
#                 p_phases=p_phases,
#                 s_phases=s_phases,
#                 key=("X", "lp_detections"),
#                 sigma=self.sigma,
#             )
#             rg_detection_labeller = EventTypeDetectionLabeller(
#                 source_type="regular",
#                 p_phases=p_phases,
#                 s_phases=s_phases,
#                 key=("X", "rg_detections"),
#                 sigma=self.sigma,
#             )

#         block1 = [
#             # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
#             # Uses strategy variable, as padding will be handled by the random window.
#             # In 1/3 of the cases, just returns the original trace, to keep diversity high.
#             sbg.OneOf(
#                 [
#                     sbg.WindowAroundSample(
#                         list(phase_dict.keys()),
#                         samples_before=first_window_kwargs["samples_before"],
#                         windowlen=first_window_kwargs["windowlen"],
#                         selection=first_window_kwargs["selection"],
#                         strategy="variable",
#                     ),
#                     sbg.NullAugmentation(),
#                 ],
#                 probabilities=first_window_prob,
#             ),
#             sbg.RandomWindow(
#                 low=self.sample_boundaries[0],
#                 high=self.sample_boundaries[1],
#                 windowlen=6000,
#                 strategy="pad",
#             ),
#             sbg.ProbabilisticLabeller(
#                 shape=self.prob_label_shape,
#                 label_columns=phase_dict,
#                 noise_column=False,
#                 sigma=self.sigma,
#                 dim=0,
#             ),
#             rg_detection_labeller,
#             lp_detection_labeller,
#             # Normalize to ensure correct augmentation behavior
#             sbg.Normalize(
#                 detrend_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
#             ),  # "peak"/“std
#         ]
#         return block1

#     def get_joint_augmentation_block2(self):
#         block2 = [
#             sbg.ChangeDtype(np.float32, "X"),
#             sbg.ChangeDtype(np.float32, "y"),
#             sbg.ChangeDtype(np.float32, "rg_detections"),
#             sbg.ChangeDtype(np.float32, "lp_detections"),
#         ]
#         return block2

#     def set_superimposed_event_generator(
#         self, train_data_eq, train_data_noise, val_data_eq, val_data_noise
#     ):
#         block1_aug = self.get_joint_augmentation_block1(
#             {"samples_before": 3000, "windowlen": 8000, "selection": "first"}, [1, 0]
#         )
#         if len(train_data_eq) > 0:
#             if not np.all(train_data_eq["split"] == "train"):
#                 raise KeyError("'train_data_eq' should be a subset of the training set")
#             self.train_stacked_event_generator = sbg.GenericGenerator(train_data_eq)
#             self.train_stacked_event_generator.add_augmentations(block1_aug)
#         else:
#             print("There are no earthquake waveforms in the training set.")

#         if len(train_data_noise) > 0:
#             if not np.all(train_data_noise["split"] == "train"):
#                 raise KeyError("'train_data_eq' should be a subset of the training set")
#             self.train_stacked_noise_generator = sbg.GenericGenerator(train_data_noise)
#             self.train_stacked_noise_generator.add_augmentations(block1_aug)
#         else:
#             print("There are no noise waveforms in the training set.")

#         if len(val_data_eq) > 0:
#             if not np.all(val_data_eq["split"] == "dev"):
#                 raise KeyError(
#                     "'val_data_eq' should be a subset of the development set"
#                 )
#             self.val_stacked_event_generator = sbg.GenericGenerator(val_data_eq)
#             self.val_stacked_event_generator.add_augmentations(block1_aug)
#         else:
#             print("There are no earthquake waveforms in the validation set.")

#         if len(val_data_noise) > 0:
#             if not np.all(val_data_noise["split"] == "dev"):
#                 raise KeyError(
#                     "'val_data_noise' should be a subset of the development set"
#                 )
#             self.val_stacked_noise_generator = sbg.GenericGenerator(val_data_noise)
#             self.val_stacked_noise_generator.add_augmentations(block1_aug)
#         else:
#             print("There are no noise waveforms in the validation set.")

#     def get_gap_augmentation_block(self):
#         # Gaps
#         gap_augmentation_block = [
#             sbg.OneOf(
#                 [
#                     sbg.AddGap(
#                         label_keys=["y", "rg_detections", "lp_detections"], noise_id={}
#                     ),
#                     sbg.NullAugmentation(),
#                 ],
#                 [0.2, 0.8],
#             )
#         ]
#         return gap_augmentation_block

#     def get_stack_block(self, event_generator, noise_generator):
#         stack_event_augmentation_block = [
#             sbg.OneOf(
#                 [
#                     SuperimposeEvent(
#                         data_generator=event_generator,
#                         label_columns=phase_dict,
#                         inv_scale=(0.25, 4),
#                         key="X",
#                         label_key="y",
#                         detection_keys=["lp_detections", "rg_detections"],
#                         sep=200,
#                         noise_label=False,
#                         noise_data=False,
#                     ),
#                     MyDuplicateEvent(
#                         label_columns=phase_dict,
#                         inv_scale=(0.25, 4),
#                         key="X",
#                         label_key="y",
#                         detection_keys=["lp_detections", "rg_detections"],
#                         sep=200,
#                         noise_label=False,
#                     ),
#                     sbg.NullAugmentation(),
#                 ],
#                 probabilities=[0.2, 0.2, 0.6],
#             ),
#             sbg.OneOf(
#                 [
#                     SuperimposeEvent(
#                         data_generator=noise_generator,
#                         label_columns=phase_dict,
#                         key="X",
#                         label_key="y",
#                         detection_keys=None,
#                         inv_scale=(1.25, 40),
#                         sep=200,
#                         noise_label=False,
#                         noise_data=True,
#                     ),
#                     sbg.GaussianNoise(),
#                     sbg.NullAugmentation(),
#                 ],
#                 probabilities=[0.25, 0.25, 0.5],
#             ),
#         ]
#         stack_event_augmentation_block = []
#         if event_generator is not None:
#             stack_event_augmentation_block.append(
#                 sbg.OneOf(
#                     [
#                         SuperimposeEvent(
#                             data_generator=event_generator,
#                             label_columns=phase_dict,
#                             inv_scale=(0.25, 4),
#                             key="X",
#                             label_key="y",
#                             detection_keys=["lp_detections", "rg_detections"],
#                             sep=200,
#                             noise_label=False,
#                             noise_data=False,
#                         ),
#                         MyDuplicateEvent(
#                             label_columns=phase_dict,
#                             inv_scale=(0.25, 4),
#                             key="X",
#                             label_key="y",
#                             detection_keys=["lp_detections", "rg_detections"],
#                             sep=200,
#                             noise_label=False,
#                         ),
#                         sbg.NullAugmentation(),
#                     ],
#                     probabilities=[0.2, 0.2, 0.6],
#                 )
#             )
#         if noise_generator is not None:
#             stack_event_augmentation_block.append(
#                 sbg.OneOf(
#                     [
#                         SuperimposeEvent(
#                             data_generator=noise_generator,
#                             label_columns=phase_dict,
#                             key="X",
#                             label_key="y",
#                             detection_keys=None,
#                             inv_scale=(1.25, 40),
#                             sep=200,
#                             noise_label=False,
#                             noise_data=True,
#                         ),
#                         sbg.GaussianNoise(),
#                         sbg.NullAugmentation(),
#                     ],
#                     probabilities=[0.25, 0.25, 0.5],
#                 )
#             )
#         return stack_event_augmentation_block

#     def get_train_augmentations(self):
#         stack_event_augmentation_block = self.get_stack_block(
#             self.train_stacked_event_generator, self.train_stacked_noise_generator
#         )
#         block1 = self.get_joint_augmentation_block1()
#         block2 = self.get_joint_augmentation_block2()

#         gap_augmentation_block = self.get_gap_augmentation_block()
#         return (
#             block1
#             + stack_event_augmentation_block
#             + gap_augmentation_block
#             + [  # Augmentations make second normalize necessary
#                 sbg.Normalize(
#                     demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
#                 )
#             ]
#             + block2
#         )

#     def get_val_augmentations(self):
#         stack_event_augmentation_block = self.get_stack_block(
#             self.val_stacked_event_generator, self.val_stacked_noise_generator
#         )
#         block1 = self.get_joint_augmentation_block1()
#         block2 = self.get_joint_augmentation_block2()

#         gap_augmentation_block = self.get_gap_augmentation_block()
#         return (
#             block1
#             + stack_event_augmentation_block
#             + gap_augmentation_block
#             + [  # Augmentations make second normalize necessary
#                 sbg.Normalize(
#                     demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
#                 )
#             ]
#             + block2
#         )

#     def get_augmentations(self):
#         raise NotImplementedError("Use get_train/val_augmentations instead.")

#     def get_eval_augmentations(self):
#         return [
#             sbg.SteeredWindow(windowlen=6000, strategy="pad"),
#             sbg.ChangeDtype(np.float32),
#             sbg.Normalize(
#                 demean_axis=-1, amp_norm_axis=-1, amp_norm_type=self.model.norm
#             ),
#         ]

#     def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
#         x = batch["X"]
#         window_borders = batch["window_borders"]

#         rg_det_pred, lp_det_pred, p_pred, s_pred = self.model(x)

#         score_detection = torch.zeros(rg_det_pred.shape[0])
#         score_rg_detection = torch.zeros(rg_det_pred.shape[0])
#         score_lp_detection = torch.zeros(rg_det_pred.shape[0])
#         score_p_or_s = torch.zeros(rg_det_pred.shape[0])
#         score_rg_or_lp = torch.zeros(rg_det_pred.shape[0])
#         p_sample = torch.zeros(rg_det_pred.shape[0], dtype=int)
#         s_sample = torch.zeros(rg_det_pred.shape[0], dtype=int)

#         for i in range(rg_det_pred.shape[0]):
#             start_sample, end_sample = window_borders[i]
#             local_rg_det_pred = rg_det_pred[i, start_sample:end_sample]
#             local_lp_det_pred = lp_det_pred[i, start_sample:end_sample]
#             local_p_pred = p_pred[i, start_sample:end_sample]
#             local_s_pred = s_pred[i, start_sample:end_sample]

#             score_detection[i] = torch.max(
#                 torch.maximum(local_rg_det_pred, local_lp_det_pred)
#             )
#             score_rg_detection[i] = torch.max(local_rg_det_pred)
#             score_lp_detection[i] = torch.max(local_lp_det_pred)
#             score_p_or_s[i] = torch.max(local_p_pred) / torch.max(
#                 local_s_pred
#             )  # most likely P by most likely S
#             score_rg_or_lp[i] = torch.max(local_rg_det_pred) / torch.max(
#                 local_lp_det_pred
#             )

#             p_sample[i] = torch.argmax(local_p_pred)
#             s_sample[i] = torch.argmax(local_s_pred)

#         return (
#             score_detection,
#             score_p_or_s,
#             p_sample,
#             s_sample,
#             score_rg_detection,
#             score_lp_detection,
#             score_rg_or_lp,
#         )


# class EventTypeDetectionLabeller(SupervisedLabeller):
#     """
#     Create detection labels from picks for a certain type of evnets .
#     The labeler can either use fixed detection length or determine the length from the P to S time as in
#     Mousavi et al. (2020, Nature communications). In the latter case, detections range from P to S + factor * (S - P)
#     and are only annotated if both P and S phases are present.
#     All detections are represented through a boxcar time series with the same length as the input waveforms.
#     For both P and S, lists of phases can be passed of which the sequentially first one will be used.
#     All picks with NaN sample are treated as not present.

#     :param source_type: Type of events to label. If the source type of the input is not the specified
#                         one, labels are zeros. Otherwise, label the detection from picks.
#     :type source_type: str
#     :param p_phases: (List of) P phase metadata columns
#     :type p_phases: str, list[str]
#     :param s_phases: (List of) S phase metadata columns
#     :type s_phases: str, list[str]
#     :param factor: Factor for length of window after S onset
#     :type factor: float
#     :param fixed_window: Number of samples for fixed window detections. If none, will determine length from P to S time.
#     :type fixed_window: int
#     """

#     def __init__(
#         self,
#         p_phases,
#         s_phases=None,
#         factor=1.4,
#         fixed_window=None,
#         source_type="lp",
#         sigma=20,
#         **kwargs,
#     ):
#         if source_type not in ["lp", "regular"]:
#             raise ValueError(f"Illegal source type (source_type={source_type}).")
#         self.label_method = "probabilistic"
#         if source_type == "lp":
#             self.label_columns = "lp_detections"
#         elif source_type == "regular":
#             self.label_columns = "rg_detections"

#         if isinstance(p_phases, str):
#             self.p_phases = [p_phases]
#         else:
#             self.p_phases = p_phases

#         if isinstance(s_phases, str):
#             self.s_phases = [s_phases]
#         elif s_phases is None:
#             self.s_phases = []
#         else:
#             self.s_phases = s_phases

#         if s_phases is not None and fixed_window is not None:
#             seisbench.logger.warning(
#                 "Provided both S phases and fixed window length to DetectionLabeller. "
#                 "Will use fixed window size and ignore S phases."
#             )

#         self.source_type = source_type

#         self.sigma = sigma
#         self.factor = factor
#         self.fixed_window = fixed_window

#         kwargs["dim"] = kwargs.get("dim", -2)
#         super().__init__(label_type="multi_class", **kwargs)

#     def label(self, X, metadata):
#         if "source_type" not in metadata:
#             raise KeyError('Training dataset does not contain "source_type"')
#         sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
#             seisbench.config, self.ndim
#         )
#         source_type_hit = False
#         if self.source_type == "lp" and metadata["source_type"] == "lp":
#             source_type_hit = True
#         elif self.source_type == "regular" and metadata["source_type"] != "lp":
#             source_type_hit = True

#         if self.fixed_window:
#             # Only label until end of fixed window
#             factor = 0
#         else:
#             factor = self.factor

#         if self.ndim == 2:
#             y = np.zeros((1, X.shape[width_dim]))
#             if source_type_hit:
#                 p_arrivals = [
#                     metadata[phase]
#                     for phase in self.p_phases
#                     if phase in metadata and not np.isnan(metadata[phase])
#                 ]
#                 if self.fixed_window is not None:
#                     # Fake S arrivals for simulating fixed window
#                     s_arrivals = [x + self.fixed_window for x in p_arrivals]
#                 else:
#                     s_arrivals = [
#                         metadata[phase]
#                         for phase in self.s_phases
#                         if phase in metadata and not np.isnan(metadata[phase])
#                     ]

#                 if len(p_arrivals) != 0 and len(s_arrivals) != 0:
#                     p_arrival = min(p_arrivals)
#                     s_arrival = min(s_arrivals)
#                     p_to_s = s_arrival - p_arrival
#                     if s_arrival >= p_arrival:
#                         # Only annotate valid options
#                         p0 = max(int(p_arrival), 0)
#                         p1 = max(int(s_arrival + factor * p_to_s), 0)
#                         y[0, p0:p1] = 1
#                         idxs_left = np.arange(0, min(p0, y.shape[1]))
#                         y[0, idxs_left] = np.exp(
#                             -np.power(idxs_left - p0, 2.0)
#                             / (2 * np.power(self.sigma, 2.0))
#                         )
#                         idxs_right = np.arange(min(p1, y.shape[1]), y.shape[1])
#                         y[0, idxs_right] = np.exp(
#                             -np.power(idxs_right - p1, 2.0)
#                             / (2.0 * np.power(self.sigma, 2.0))
#                         )

#         elif self.ndim == 3:
#             y = np.zeros(
#                 shape=(
#                     X.shape[sample_dim],
#                     1,
#                     X.shape[width_dim],
#                 )
#             )
#             if source_type_hit:
#                 p_arrivals = [
#                     metadata[phase] for phase in self.p_phases if phase in metadata
#                 ]

#                 if self.fixed_window is not None:
#                     # Fake S arrivals for simulating fixed window
#                     s_arrivals = [x + self.fixed_window for x in p_arrivals]
#                 else:
#                     s_arrivals = [
#                         metadata[phase] for phase in self.s_phases if phase in metadata
#                     ]

#                 if len(p_arrivals) != 0 and len(s_arrivals) != 0:
#                     p_arrivals = np.stack(
#                         p_arrivals, axis=-1
#                     )  # Shape (samples, phases)
#                     s_arrivals = np.stack(s_arrivals, axis=-1)

#                     mask = np.logical_and(
#                         np.any(~np.isnan(p_arrivals), axis=1),
#                         np.any(~np.isnan(s_arrivals), axis=1),
#                     )
#                     if not mask.any():
#                         return y

#                     p_arrivals = np.nanmin(
#                         p_arrivals[mask, :], axis=1
#                     )  # Shape (samples (which are present),)
#                     s_arrivals = np.nanmin(s_arrivals[mask, :], axis=1)
#                     p_to_s = s_arrivals - p_arrivals

#                     starts = p_arrivals.astype(int)
#                     ends = (s_arrivals + factor * p_to_s).astype(int)

#                     # print(mask, starts, ends)
#                     for i, s, e in zip(np.arange(len(mask))[mask], starts, ends):
#                         s = max(0, s)
#                         e = max(0, e)
#                         y[i, 0, s:e] = 1

#         else:
#             raise ValueError(
#                 f"Illegal number of input dimensions for DetectionLabeller (ndim={self.ndim})."
#             )

#         return y

#     def __str__(self):
#         return f"DetectionLabeller (label_type={self.label_type}, dim={self.dim})"
