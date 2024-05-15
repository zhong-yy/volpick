"""
This file contains the following customized augmentaions:
(1) Stack the waveform of another event on the current event. The stacked event is randomly selected 
    from a data generator. 
(2) Stack a rescaled and shifted copy of the current trace on itself.

Acknowledgement: The implementation is inspired by and modified from Jannes Münchmeyer's implementation 
of duplicating events: https://github.com/seisbench/pick-benchmark/blob/main/benchmark/augmentations.py.
"""

import numpy as np
import copy
import seisbench.generate as sbg


def label_normalization_method1(y, phase_ids, noise_id):
    y[phase_ids, :] /= np.maximum(1, np.nansum(y[phase_ids, :], axis=0, keepdims=True))
    y[noise_id, :] = 1 - np.sum(y[phase_ids, :], 0)


def label_normalization_method2(y, phase_ids, noise_id):
    prob_sum = np.sum(y[phase_ids, :], axis=0)
    prob_inter = np.prod(y[phase_ids, :], axis=0)

    for pha_lab_id in phase_ids:
        frac = y[pha_lab_id, :] / np.maximum(prob_sum, 1e-6)
        y[pha_lab_id, :] = y[pha_lab_id, :] - prob_inter + prob_inter * frac
    y[noise_id, :] = 1 - np.sum(y[phase_ids, :], 0)


class SuperimposeEvent:
    """
    Superimpose a rescaled event waveform trace randomly from a given generator on the current trace.
    The superimposed event is randomly placed between the end of the current event (determined by S phase)
    the end of the whole trace.

    .. warning::
        This augmentation does **not** modify the metadata, as representing multiple picks of
        the same type is currently not supported. Workflows should therefore always first generate
        labels from metadata and then pass the labels in the key `label_keys`. These keys are automatically
        adjusted by addition of the labels.

    .. warning::
        This implementation currently has strict shape requirements:

        - (1, samples) for detection
        - (channels, samples) for data
        - (labels, samples) for labels

    :param inv_scale: The scale factor is defined by as 1/u, where u is uniform.
                      `inv_scale` defines the minimum and maximum values for u.
                      Defaults to (1, 10), e.g., scaling by factor 1 to 1/10.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key
                to read from and the second one the key to write to.
    :param label_keys: Keys for the label columns.
                       Labels of the original and duplicate events will be added and capped at 1.
                       Note that this will lead to invalid noise traces.
                       Value can either be a single key specification or a list of key specifications.
                       Each key specification is either a string, for identical input and output keys,
                       or as a tuple of two strings, input and output keys.
                       Defaults to None.
    :param detection_key
    :param noise_label
    :param noise_data
    :param prob_num_events
    """

    def __init__(
        self,
        data_generator,
        label_columns,
        inv_scale=(1, 10),
        key="X",
        label_key="y",
        detection_keys=["detections", "lp_detections", "rg_detections"],
        noise_label=True,
        noise_data=False,
        prob_num_events={1: 0.7, 2: 0.3},
        sep=20,
        tail_length_factor=1.4,
    ):
        self.key = (key, key)
        self.label_key = (label_key, label_key)
        assert self.key[0] == self.key[1]
        assert self.label_key[0] == self.label_key[1]

        # detection
        if not isinstance(detection_keys, list):
            if detection_keys is None:
                detection_keys = []
            else:
                detection_keys = [detection_keys]

        self.detection_keys = []
        for key in detection_keys:
            if isinstance(key, tuple):
                self.detection_keys.append(key)
            else:
                self.detection_keys.append((key, key))

        # if detection_key is not None:
        #     self.detection_key = (detection_key, detection_key)
        #     # assert self.detection_key[0] == self.detection_key[1]
        # else:
        #     self.detection_key = None

        # if lp_detection_key is not None:
        #     self.lp_detection_key = (lp_detection_key, lp_detection_key)
        #     # assert self.lp_detection_key[0] == self.lp_detection_key[1]
        # else:
        #     self.lp_detection_key = None

        # if rg_detection_key is not None:
        #     self.rg_detection_key = (rg_detection_key, rg_detection_key)
        #     # assert self.rg_detection_key[0] == self.rg_detection_key[1]
        # else:
        #     self.rg_detection_key = None

        self.inv_scale = inv_scale
        self.data_generator = data_generator
        self.sep = sep

        self.label_columns = label_columns
        (
            self.label_columns,
            self.labels,
            self.label_ids,
        ) = sbg.PickLabeller._columns_to_dict_and_labels(
            label_columns, noise_column=noise_label
        )
        self.noise_label = noise_label

        self.noise_data = noise_data
        self.ev_num_choices = []
        self.ev_num_weights = []
        for n_choice, n_weight in prob_num_events.items():
            self.ev_num_choices.append(n_choice)
            self.ev_num_weights.append(n_weight)

        self.phase_ids = [
            self.label_ids[label] for label in self.label_ids.keys() if label != "Noise"
        ]

        self.tail_length_factor = tail_length_factor

    def __call__(self, state_dict):
        n_secondary_events = np.random.choice(
            self.ev_num_choices, p=self.ev_num_weights
        )
        if self.noise_data:
            for i in range(n_secondary_events):
                x, metadata = state_dict[self.key[0]]
                idx = np.random.randint(len(self.data_generator))
                second_waveform = self.data_generator[idx]
                scale = 1 / np.random.uniform(*self.inv_scale) * np.max(np.abs(x))
                x2 = second_waveform[self.key[0]]
                for cha_idx in range(x.shape[0]):
                    if np.allclose(x[cha_idx], 0):
                        x2[cha_idx] = 0
                x = x + scale * x2
                # print(scale)
                state_dict[self.key[1]] = (x, metadata)
                # state_dict["noise"] = (scale * x2, copy.deepcopy(metadata))
        else:
            x, metadata = state_dict[self.key[0]]
            n_samples = x.shape[-1]

            onsets = []
            for label_column, label in self.label_columns.items():
                if label_column not in metadata:
                    # Unknown pick
                    continue

                if isinstance(
                    metadata[label_column], (int, np.integer, float)
                ) and not np.isnan(metadata[label_column]):
                    onsets.append(metadata[label_column])
            if len(onsets) == 0:
                first_event_end = 0
                return  # do not stack anything on a noise trace
            elif len(onsets) >= 2:
                first_event_end = int(
                    max(onsets)
                    + max(
                        (max(onsets) - min(onsets)) * self.tail_length_factor,
                        self.sep,
                    )
                    + 0.2 * self.sep
                )
            else:
                first_event_end = int(max(onsets) + 1 + self.sep)
            x[:, min(x.shape[1], int(first_event_end)) :] = 0

            for i in range(n_secondary_events):
                if first_event_end >= n_samples - 2 * self.sep:
                    # print(i, "out of range", first_event_end)
                    break

                x, metadata = state_dict[self.key[0]]
                y, metadata = state_dict[self.label_key[0]]

                idx = np.random.randint(len(self.data_generator))
                second_waveform = self.data_generator[idx]
                x2 = second_waveform[self.key[0]]
                for cha_idx in range(x.shape[0]):
                    if np.allclose(x[cha_idx], 0):
                        x2[cha_idx] = 0
                y2 = second_waveform[self.label_key[0]]
                if not np.isclose(np.max(y2[self.label_ids["P"], :]), 1, atol=1e-2):
                    continue

                original_first_pick = np.argmax(y2[self.label_ids["P"], :])
                x2[:, : max(original_first_pick - int(self.sep), 0)] = 0
                # print(
                #     first_event_end,
                #     np.max(y2[self.label_ids["P"], :]),
                #     np.argmax(y2[self.label_ids["P"], :]),
                #     np.max(y2[self.label_ids["S"], :]),
                #     np.argmax(y2[self.label_ids["S"], :]),
                # )
                shifted_first_pick = np.random.randint(
                    first_event_end, n_samples - 2 * self.sep
                )
                # print(first_event_end, shifted_first_pick)
                shift = abs(shifted_first_pick - original_first_pick)

                x2_new = np.zeros_like(x2)
                y2_new = np.zeros_like(y2)
                if original_first_pick < shifted_first_pick:  # shift to the right
                    x2_new[:, shift:] = x2[:, :-shift]
                    y2_new[:, shift:] = y2[:, :-shift]
                elif shifted_first_pick < original_first_pick:  # shift to the left
                    x2_new[:, :-shift] = x2[:, shift:]
                    y2_new[:, :-shift] = y2[:, shift:]
                else:
                    x2_new[...] = x2[...]
                    y2_new[...] = y2[...]
                # print(
                #     "b",
                #     shift,
                #     shifted_first_pick,
                #     max(np.argmax(y2_new[self.phase_ids, :], axis=1)),
                # )

                scale = 1 / np.random.uniform(*self.inv_scale)

                x = x + scale * x2_new
                y = np.maximum(y, y2_new)

                if self.noise_label:
                    # y[phase_ids, :] /= np.maximum(
                    #     1, np.nansum(y[phase_ids, :], axis=0, keepdims=True)
                    # )
                    # y[self.label_ids["Noise"], :] = 1 - np.nansum(
                    #     y[phase_ids, :], axis=0
                    # )
                    label_normalization_method1(
                        y,
                        phase_ids=self.phase_ids,
                        noise_id=self.label_ids["Noise"],
                    )
                state_dict[self.label_key[1]] = (y, metadata)

                if self.detection_keys:
                    for detection_key in self.detection_keys:
                        detection, metadata = state_dict[detection_key[0]]
                        if detection.shape[-1] != x.shape[-1]:
                            raise ValueError(
                                "Number of samples in trace and detection disagree."
                            )
                        detection2 = second_waveform[detection_key[0]]
                        detection2_new = np.zeros_like(detection2)
                        if (
                            original_first_pick < shifted_first_pick
                        ):  # shift to the right
                            detection2_new[:, shift:] = detection2[:, :-shift]
                        elif (
                            shifted_first_pick < original_first_pick
                        ):  # shift to the left
                            detection2_new[:, :-shift] = detection2[:, shift:]
                        detection = np.maximum(detection, detection2_new)
                        state_dict[detection_key[1]] = (detection, metadata)
                # if self.detection_key is not None:
                #     detection, metadata = state_dict[self.detection_key[0]]
                #     if detection.shape[-1] != x.shape[-1]:
                #         raise ValueError(
                #             "Number of samples in trace and detection disagree."
                #         )
                #     detection2 = second_waveform[self.detection_key[0]]
                #     detection2_new = np.zeros_like(detection2)
                #     if original_first_pick < shifted_first_pick:  # shift to the right
                #         detection2_new[:, shift:] = detection2[:, :-shift]
                #     elif shifted_first_pick < original_first_pick:  # shift to the left
                #         detection2_new[:, :-shift] = detection2[:, shift:]
                #     detection = np.maximum(detection, detection2_new)
                #     state_dict[self.detection_key[1]] = (detection, metadata)

                state_dict[self.key[1]] = (x, metadata)
                if i != n_secondary_events - 1:
                    first_event_end = max(
                        first_event_end,
                        int(
                            max(np.argmax(y2_new[self.phase_ids, :], axis=1))
                            + 1
                            + self.sep
                        ),
                    )


class MyDuplicateEvent:
    """
    Superimpose a shifted copy of a trace on itself
    .. warning::
        This augmentation does **not** modify the metadata, as representing multiple picks of
        the same type is currently not supported. Workflows should therefore always first generate
        labels from metadata and then pass the labels in the key `label_keys`. These keys are automatically
        adjusted by addition of the labels.

    .. warning::
        This implementation currently has strict shape requirements:

        - (1, samples) for detection
        - (channels, samples) for data
        - (labels, samples) for labels

    :param inv_scale: The scale factor is defined by as 1/u, where u is uniform.
                      `inv_scale` defines the minimum and maximum values for u.
                      Defaults to (1, 10), e.g., scaling by factor 1 to 1/10.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key
                to read from and the second one the key to write to.
    :param label_keys: Keys for the label columns.
                       Labels of the original and duplicate events will be added and capped at 1.
                       Note that this will lead to invalid noise traces.
                       Value can either be a single key specification or a list of key specifications.
                       Each key specification is either a string, for identical input and output keys,
                       or as a tuple of two strings, input and output keys.
                       Defaults to None.
    :param noise_id
    """

    def __init__(
        self,
        label_columns,
        inv_scale=(1, 10),
        key="X",
        label_key="y",
        detection_keys=["detections", "lp_detections", "rg_detections"],
        noise_label=True,
        prob_num_events={1: 0.7, 2: 0.3},
        sep=20,
        tail_length_scale=1.4,
    ):
        self.key = (key, key)
        self.label_key = (label_key, label_key)
        assert self.key[0] == self.key[1]
        assert self.label_key[0] == self.label_key[1]
        # if detection_key is not None:
        #     self.detection_key = (detection_key, detection_key)
        #     assert self.detection_key[0] == self.detection_key[1]
        # else:
        #     self.detection_key = detection_key

        # detection
        if not isinstance(detection_keys, list):
            if detection_keys is None:
                detection_keys = []
            else:
                detection_keys = [detection_keys]

        self.detection_keys = []
        for key in detection_keys:
            if isinstance(key, tuple):
                self.detection_keys.append(key)
            else:
                self.detection_keys.append((key, key))
        self.inv_scale = inv_scale
        self.sep = sep

        self.label_columns = label_columns
        (
            self.label_columns,
            self.labels,
            self.label_ids,
        ) = sbg.PickLabeller._columns_to_dict_and_labels(
            label_columns, noise_column=noise_label
        )
        self.noise_label = noise_label

        # self.prob_num_events = prob_num_events
        self.ev_num_choices = []
        self.ev_num_weights = []
        for n_choice, n_weight in prob_num_events.items():
            self.ev_num_choices.append(n_choice)
            self.ev_num_weights.append(n_weight)

        self.phase_ids = [
            self.label_ids[label] for label in self.label_ids.keys() if label != "Noise"
        ]
        self.tail_length_factor = tail_length_scale

    def __call__(self, state_dict):
        n_secondary_events = np.random.choice(
            self.ev_num_choices, p=self.ev_num_weights
        )
        x, metadata = state_dict[self.key[0]]
        y, _ = state_dict[self.label_key[0]]
        x2 = x.copy()
        y2 = y.copy()
        if not np.isclose(np.max(y2[self.label_ids["P"], :]), 1, atol=1e-2):
            return

        if self.detection_keys:
            detection_labels = {}
            for detection_key in self.detection_keys:
                detection, _ = state_dict[detection_key[0]]
                # detection2 = detection.copy()
                detection_labels[detection_key] = detection.copy()

        n_samples = x.shape[-1]

        onsets = []
        for label_column, label in self.label_columns.items():
            if label_column not in metadata:
                # Unknown pick
                continue

            if isinstance(
                metadata[label_column], (int, np.integer, float)
            ) and not np.isnan(metadata[label_column]):
                onsets.append(metadata[label_column])
        if len(onsets) == 0:
            # first_event_start = 0
            first_event_end = 0
            return  # do not stack anything on a noise trace
        elif len(onsets) >= 2:
            first_event_end = int(
                max(onsets)
                + max(
                    (max(onsets) - min(onsets)) * self.tail_length_factor,
                    self.sep,
                )
                + 0.2 * self.sep
            )
        else:
            # first_event_start = min(onsets) - 1 - self.sep
            first_event_end = max(onsets) + 1 + self.sep
        x[:, min(x.shape[1], int(first_event_end)) :] = 0
        # length = first_event_end - first_event_start

        original_first_pick = np.argmax(y2[self.label_ids["P"], :])
        x2[:, : max(original_first_pick - int(self.sep), 0)] = 0
        for i in range(n_secondary_events):
            if first_event_end + 2 * self.sep >= n_samples:
                break

            x, metadata = state_dict[self.key[0]]
            y, metadata = state_dict[self.label_key[0]]
            # if y.shape[-1] != n_samples:
            #     raise ValueError(
            #         f"Number of samples disagree between trace and label key '{self.label_key[0]}'."
            #     )

            # print(
            #     first_event_end,
            #     np.max(y2[self.label_ids["P"], :]),
            #     np.argmax(y2[self.label_ids["P"], :]),
            #     np.max(y2[self.label_ids["S"], :]),
            #     np.argmax(y2[self.label_ids["S"], :]),
            # )
            shifted_first_pick = np.random.randint(
                first_event_end, n_samples - self.sep
            )
            shift = abs(shifted_first_pick - original_first_pick)
            # print(first_event_end, shifted_first_pick)
            # print(shifted_first_pick)

            x2_new = np.zeros_like(x2)
            y2_new = np.zeros_like(y2)
            if original_first_pick < shifted_first_pick:  # shift to the right
                x2_new[:, shift:] = x2[:, :-shift]
                y2_new[:, shift:] = y2[:, :-shift]
            elif shifted_first_pick < original_first_pick:  # shift to the left
                x2_new[:, :-shift] = x2[:, shift:]
                y2_new[:, :-shift] = y2[:, shift:]
            else:
                x2_new[...] = x2[...]
                y2_new[...] = y2[...]

            # print(
            #     "b",
            #     shift,
            #     shifted_first_pick,
            #     max(np.argmax(y2_new[self.phase_ids, :], axis=1)),
            # )

            scale = 1 / np.random.uniform(*self.inv_scale)
            x = x + scale * x2_new
            y = np.maximum(y, y2_new)

            if self.noise_label:
                # y[phase_ids, :] /= np.maximum(
                #     1, np.nansum(y[phase_ids, :], axis=0, keepdims=True)
                # )
                # y[self.label_ids["Noise"], :] = 1 - np.nansum(
                #     y[phase_ids, :], axis=0
                # )
                label_normalization_method1(
                    y,
                    phase_ids=self.phase_ids,
                    noise_id=self.label_ids["Noise"],
                )
            state_dict[self.label_key[1]] = (y, metadata)

            if self.detection_keys:
                for detection_key in self.detection_keys:
                    detection, metadata = state_dict[detection_key[0]]
                    if detection.shape[-1] != x.shape[-1]:
                        raise ValueError(
                            "Number of samples in trace and detection disagree."
                        )
                    detection2 = detection_labels[detection_key]
                    detection2_new = np.zeros_like(detection2)
                    if original_first_pick < shifted_first_pick:  # shift to the right
                        detection2_new[:, shift:] = detection2[:, :-shift]
                    elif shifted_first_pick < original_first_pick:  # shift to the left
                        detection2_new[:, :-shift] = detection2[:, shift:]
                    detection = np.maximum(detection, detection2_new)
                    state_dict[detection_key[1]] = (detection, metadata)
            state_dict[self.key[1]] = (x, metadata)
            if i != n_secondary_events - 1:
                first_event_end = max(
                    first_event_end,
                    max(np.argmax(y2_new[self.phase_ids, :], axis=1)) + 1 + self.sep,
                )
                # first_event_end = (
                #     first_event_end + shifted_first_pick - original_first_pick
                # )
