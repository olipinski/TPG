"""
Dataset utilities for the Temporal Progression Games.

Generates the required datasets for the environment to use.
"""
import copy
import warnings
from itertools import chain
from typing import Optional

import numpy as np
from numpy.random import default_rng
from torch.utils import data


class ProgressiveDataset(data.Dataset):
    """
    The class that implements dataset generation for Temporal Progression Games.

    This can generate the data required for the environment,
    with different kinds being generated for the given dataset_type.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        dataset_size: int = 10000,
        num_points: int = 100,
        num_distractors: int = 3,
        repeat_chance: float = 0.0,
        sequence_window: bool = True,
        sequence_window_size: int = 2,
        use_random: bool = True,
        generate_special: dict = None,
    ):
        """
        Initialise the dataset class, which also generates the dataset.

        Parameters
        ----------
        seed: int
            Random seed
        dataset_size: int
            Size of the dataset
        num_points: int
            Number of points in the dataset
        num_distractors: int
            Number of distractors in the dataset
        repeat_chance: float
            Chance of repetitions of the targets
        sequence_window: bool
            Whether to use a sequence window, or show whole sequence to sender
        sequence_window_size: int
            The size of the sequence window, on each side of the target.
        use_random: bool
            Whether to use a random sequence at every target, or to use a [1...N] sequence
        generate_special: dict
            Whether to generate a special dataset, containing only the targets that the messages can correspond to.
            The special dataset will be generated only for messages provided in this dict.
        """
        self.seed = seed
        self.dataset_size = dataset_size
        self.num_points = num_points
        self.num_distractors = num_distractors
        self.repeat_chance = repeat_chance
        self.sequence_window = sequence_window
        self.sequence_window_size = sequence_window_size
        self.use_random = use_random
        self.generate_special = generate_special
        self.compo_failure_counter = 0

        if self.generate_special is not None:
            self.to_generate = []
            self.c_pos = False
            if "nc" in self.generate_special:
                self.to_generate += list(self.generate_special["nc"].keys())
                self.nc = True  # Generate non-compositional
            if "pos" in self.generate_special:
                self.to_generate += list(self.generate_special["pos"])
            if "c-int" in self.generate_special:
                if "c-pos" in self.generate_special:
                    # c-pos is mutually exclusive with c-int
                    self.to_generate += ["c-pos"]
                else:
                    self.to_generate += ["c-int"]

        self.rng = default_rng(seed)

        if sequence_window:
            if sequence_window_size * 2 > num_points:
                # Cannot window as too few points
                raise ValueError("Sequence window size too large!")

        (
            self.sequences,
            self.cut_inputs,
            self.tds,
            self.target_labels,
        ) = self._generate_dataset()

        if self.compo_failure_counter > 0:
            warnings.warn(
                f"Compositional dataset creation failure counter is not 0, but {self.compo_failure_counter}."
                f" If this is very high, the results may be inaccurate."
                f" Percentage of total dataset {self.compo_failure_counter/self.dataset_size}"
            )

    def __getitem__(self, index: int):
        """
        Return the batch item with given index.

        Parameters
        ----------
        index: int
            Index of the item to return.

        Returns
        -------
        item
        """
        return (
            self.sequences[index],
            self.cut_inputs[index],
            self.tds[index],
            self.target_labels[index],
        )

    def __len__(self) -> int:
        """
        Return the length of the dataset, which is the number of objects contained within.

        Returns
        -------
        len: int
            Number of objects in the dataset.
        """
        return self.dataset_size

    def _generate_dataset(self):
        cut_inputs = []
        targets_and_distractorss = []
        target_labels = []
        sequences = []

        for i in range(self.dataset_size):
            if self.use_random:
                sequence = self.rng.choice(
                    self.num_points, size=self.num_points, replace=False
                )
            else:
                sequence = np.array(range(0, self.num_points, 1))

            # This contains missing items and distractors
            object_ids = self.rng.choice(
                self.num_points, self.num_distractors + 1, False
            )
            if self.generate_special:
                object_ids, sequence = self._generate_special_helper(
                    object_ids, sequence
                )
            cut_input = copy.deepcopy(sequence)
            cut_input[object_ids[0]] = -1  # Missing items set to -1
            target_and_distractors = sequence[object_ids]

            # Should the agent only see a window into the sequence?
            if self.sequence_window:
                # Make sure not over sequence boundaries
                # If would go over, then extend the other way

                # Objects would go under left side
                if object_ids[0] - self.sequence_window_size < 0:
                    under = abs(object_ids[0] - self.sequence_window_size)
                    cut_input = cut_input[
                        object_ids[0]
                        - self.sequence_window_size
                        + under : object_ids[0]
                        + self.sequence_window_size
                        + 1
                        + under
                    ]
                # Objects would go over right side
                elif object_ids[0] + self.sequence_window_size > self.num_points - 1:
                    over = abs(
                        object_ids[0] + self.sequence_window_size - self.num_points + 1
                    )
                    cut_input = cut_input[
                        object_ids[0]
                        - self.sequence_window_size
                        - over : object_ids[0]
                        + self.sequence_window_size
                        - over
                        + 1
                    ]
                else:
                    cut_input = cut_input[
                        object_ids[0]
                        - self.sequence_window_size : object_ids[0]
                        + self.sequence_window_size
                        + 1
                    ]

            # Maybe repeat the previous target?
            repeat = False
            if self.repeat_chance >= self.rng.random():
                # Repeat target if possible
                if i > 0:
                    # Repeat possible
                    cut_input = copy.deepcopy(cut_inputs[i - 1])
                    target_and_distractors = copy.deepcopy(
                        targets_and_distractorss[i - 1]
                    )
                    sequence = sequences[i - 1]
                    repeat = True
                else:
                    # Repeat impossible so we just continue
                    pass

            # Shuffle targets and distractors since the code above always uses index 0 for generation
            self.rng.shuffle(target_and_distractors)

            cut_inputs.append(cut_input)
            targets_and_distractorss.append(target_and_distractors)
            if not repeat:
                target_labels.append(
                    np.where(target_and_distractors == sequence[object_ids[0]])[0]
                )
            else:
                target_labels.append(
                    np.where(
                        target_and_distractors
                        == targets_and_distractorss[i - 1][target_labels[i - 1]]
                    )[0]
                )

            sequences.append(sequence)

        return sequences, cut_inputs, targets_and_distractorss, target_labels

    def _generate_special_helper(self, object_ids, sequence):
        special = self.rng.choice(self.to_generate)
        match special:
            case "begin":
                # Lucky, object id is already begin
                if object_ids[0] == 0:
                    pass
                # Begin was a distractor, lets swap
                elif np.any(object_ids == 0):
                    idx = np.where(object_ids == 0)[0][0]
                    object_ids[idx] = object_ids[0]
                    object_ids[0] = 0
                # Add begin as target
                else:
                    object_ids[0] = 0
            case "begin+1":
                # Lucky, object id is already begin+1
                if object_ids[0] == 1:
                    pass
                # Begin+1 was a distractor, lets swap
                elif np.any(object_ids == 1):
                    idx = np.where(object_ids == 1)[0][0]
                    object_ids[idx] = object_ids[0]
                    object_ids[0] = 1
                # Add begin+1 as target
                else:
                    object_ids[0] = 1
            case "end-1":
                # Lucky, object id is already end-1
                if object_ids[0] == self.num_points - 2:
                    pass
                # End-1 was a distractor, lets swap
                elif np.any(object_ids == self.num_points - 2):
                    idx = np.where(object_ids == self.num_points - 2)[0][0]
                    object_ids[idx] = object_ids[0]
                    object_ids[0] = self.num_points - 2
                # Add end-1 as target
                else:
                    object_ids[0] = self.num_points - 2
            case "end":
                # Lucky, object id is already end
                if object_ids[0] == self.num_points - 1:
                    pass
                # End was a distractor, lets swap
                elif np.any(object_ids == self.num_points - 1):
                    idx = np.where(object_ids == self.num_points - 1)[0][0]
                    object_ids[idx] = object_ids[0]
                    object_ids[0] = self.num_points - 1
                # Add end as target
                else:
                    object_ids[0] = self.num_points - 1
            case "c-int":
                # Generate compositional
                mappable_ints = []
                mappable_ints_count = 0
                for pos in self.generate_special["c-int"]:
                    if "inv_npmi" in pos:
                        inv_ints = list(
                            self.generate_special["c-int"]["inv_npmi"].keys()
                        )
                        mappable_ints.append(inv_ints)
                        mappable_ints_count += len(inv_ints)
                    else:
                        ints = list(self.generate_special["c-int"][pos].keys())
                        mappable_ints.append(ints)
                        # These have only one valid spot in the message, so can only actually map one int
                        mappable_ints_count += 1

                # Flatten the list
                mappable_ints = np.array(list(chain.from_iterable(mappable_ints)))
                mappable_ints = list(set(mappable_ints))

                # We can map at most 3 ints anyway
                if mappable_ints_count > 3:
                    mappable_ints_count = 3

                # If there are ngram repeats, the invariant calculation can overestimate the number of mappable ints
                if mappable_ints_count > len(mappable_ints):
                    mappable_ints_count = len(mappable_ints)

                # Choose a random X ints to map into the observation
                integers_to_obs = self.rng.choice(
                    mappable_ints, size=mappable_ints_count, replace=False
                )
                # Create new sequence without mappable ints
                new_seq = np.array(
                    [
                        x
                        for x in np.array(range(0, self.num_points, 1))
                        if x not in integers_to_obs
                    ]
                )
                self.rng.shuffle(new_seq)
                # Get random positions in the obs of where to put the mappable ints
                integers_to_obs_pos = self.rng.choice(
                    [0, 1, 3, 4], size=mappable_ints_count, replace=False
                )
                # Starting position has to be for the smaller new_seq at first
                # So we need to cut it down by 5 and the number of ints that should be mappable
                starting_pos = self.rng.choice(
                    [
                        x
                        for x in np.array(
                            range(2, self.num_points - (5 + mappable_ints_count), 1)
                        )
                    ]
                )  # Has to start before end of the shorter array!
                # New target is in the middle
                new_target = starting_pos + 2
                integers_to_obs_pos += starting_pos
                # Replace the old ints with new ints for that position
                old_ints = []
                for new_pos, integer in enumerate(integers_to_obs):
                    old_ints.append(new_seq[integers_to_obs_pos[new_pos]])
                    new_seq[integers_to_obs_pos[new_pos]] = integer

                # Append the old ints back to the array
                new_seq = np.append(new_seq, old_ints)

                sequence = new_seq
                # Lucky, object id is already correct
                if object_ids[0] == new_target:
                    pass
                # End was a distractor, lets swap
                elif np.any(object_ids == new_target):
                    idx = np.where(object_ids == new_target)[0][0]
                    object_ids[idx] = object_ids[0]
                    object_ids[0] = new_target
                # Add end as target
                else:
                    object_ids[0] = new_target
            case "c-pos":
                found, pos_int = self._compo_positional_helper()
                if found:
                    integer_pos = pos_int[0]
                    target_int = pos_int[1]
                    object_ids, sequence = self._non_compo_int_helper(
                        target_int, integer_pos, object_ids
                    )
                else:
                    self.compo_failure_counter += 1

            case _:
                ints = self.generate_special["nc"][special].keys()
                target_int = self.rng.choice(list(ints))
                object_ids, sequence = self._non_compo_int_helper(
                    target_int, special, object_ids
                )

        return object_ids, sequence

    def _compo_positional_helper(self):
        all_pos = ["npmi_pos_0", "npmi_pos_1", "npmi_pos_2"]
        referent_poses_msg = list(self.generate_special["c-pos"].keys())
        integer_poses_msg = list(self.generate_special["c-int"].keys())
        # Shuffle so its not always just 0
        self.rng.shuffle(referent_poses_msg)
        self.rng.shuffle(integer_poses_msg)
        # how to do?
        # shuffle valid ref positions list
        # sequentially go through it
        # if int found then cool, create
        # if no int found continue
        # if ended without then how?
        # random sequence and object i guess?
        # have a counter and report is as a warning if > 0,
        # so we know if its very high the result is rubbish
        for referent_pos_msg in referent_poses_msg:
            # Find valid positions for integers
            if referent_pos_msg == "inv_npmi":
                valid_int_poses = all_pos
            else:
                valid_int_poses = [item for item in all_pos if item != referent_pos_msg]

            referent_position_keys = list(
                self.generate_special["c-pos"][referent_pos_msg].keys()
            )
            self.rng.shuffle(referent_position_keys)
            # Choose random referent positions, eg "l1".

            for referent_position_pos in referent_position_keys:
                # Go through all possible ngrams randomly
                ngrams = self.generate_special["c-pos"][referent_pos_msg][
                    referent_position_pos
                ]
                self.rng.shuffle(ngrams)
                # Find length of positional reference,
                # since we will need to prune valid positions for the integers
                for ngram in ngrams:
                    max_len = 2
                    if len(ngram) == 2:
                        max_len = 1
                        # Remove the impossible positions from possible positions for the integer ngram
                        match referent_pos_msg:
                            case "inv_npmi":
                                # Positional reference will always take the middle spot
                                valid_int_poses = ["npmi_pos_0", "npmi_pos_2"]
                            case "npmi_pos_0":
                                valid_int_poses = ["npmi_pos_2"]
                            case "npmi_pos_1":
                                valid_int_poses = ["npmi_pos_0"]
                            case "npmi_pos_2":
                                raise ValueError("n-gram of len 2 cannot be in pos 2")
                    # We have a message position now
                    # And a referent position
                    # time to find an integer that can fit into this kind of message
                    for integer_pos_msg in integer_poses_msg:
                        # Can only be in valid poses or invariant, then we can (maybe) find a spot
                        if (
                            integer_pos_msg not in valid_int_poses
                            and integer_pos_msg != "inv_npmi"
                        ):
                            continue
                        # Randomly pick an integer as well
                        possible_integers = list(
                            self.generate_special["c-int"][integer_pos_msg].keys()
                        )
                        self.rng.shuffle(possible_integers)
                        for possible_integer in possible_integers:
                            num_ngrams = len(
                                self.generate_special["c-int"][integer_pos_msg][
                                    possible_integer
                                ]
                            )
                            if num_ngrams > 1:
                                for ngram in self.generate_special["c-int"][
                                    integer_pos_msg
                                ][possible_integer]:
                                    ngram_len = len(ngram)
                                    if ngram_len == 2:
                                        match integer_pos_msg:
                                            case "npmi_pos_0":
                                                if "npmi_pos_1" not in valid_int_poses:
                                                    continue
                                            case "npmi_pos_1":
                                                if "npmi_pos_2" not in valid_int_poses:
                                                    continue
                                            case "npmi_pos_2":
                                                raise ValueError(
                                                    "n-gram of len 2 cannot be in pos 2"
                                                )
                                    if ngram_len <= max_len:
                                        return True, [
                                            referent_position_pos,
                                            possible_integer,
                                        ]
                            else:
                                ngram = self.generate_special["c-int"][integer_pos_msg][
                                    possible_integer
                                ][0]
                                ngram_len = len(ngram)
                                if ngram_len == 2:
                                    match integer_pos_msg:
                                        case "npmi_pos_0":
                                            if "npmi_pos_1" not in valid_int_poses:
                                                continue
                                        case "npmi_pos_1":
                                            if "npmi_pos_2" not in valid_int_poses:
                                                continue
                                        case "npmi_pos_2":
                                            raise ValueError(
                                                "n-gram of len 2 cannot be in pos 2"
                                            )
                                if ngram_len <= max_len:
                                    return True, [
                                        referent_position_pos,
                                        possible_integer,
                                    ]

        # If not found by now, then cannot find any
        # Return not found and empty list
        return False, []

    def _non_compo_int_helper(
        self, target_int: int, int_pos: str, object_ids: np.ndarray
    ):
        new_seq = np.array(
            [x for x in np.array(range(0, self.num_points, 1)) if x != target_int]
        )
        self.rng.shuffle(new_seq)
        # The target int needs to be moved to correct position, and swapped with whatever was there
        idx = self.rng.choice(range(0, self.num_points - 1))
        if int_pos == "l":
            int_pos = self.rng.choice(list(["l1", "l2", "l3", "l4"]))
        elif int_pos == "r":
            int_pos = self.rng.choice(list(["r1", "r2", "r3", "r4"]))

        match int_pos:
            case "l1":
                # Need to have space for the new target
                if idx >= self.num_points - 1:
                    idx -= 1
                new_target = idx + 1
            case "l2":
                if idx >= self.num_points - 2:
                    idx -= 2
                new_target = idx + 2
            case "l3":
                # Must be end-1 of seq, to be truncated and extended to the left
                idx = self.num_points - 4
                new_target = idx + 3
            case "l4":
                # Must be end of seq, to be truncated and extended to the left
                idx = self.num_points - 5
                new_target = idx + 4
            case "r1":
                if idx < 1:
                    idx += 1
                new_target = idx - 1
            case "r2":
                if idx < 2:
                    idx += 2
                new_target = idx - 2
            case "r3":
                # Must be begin+1 of seq, to be truncated and extended to the right
                idx = 3
                new_target = idx - 3
            case "r4":
                # Must be begin of seq, to be truncated and extended to the right
                idx = 4
                new_target = idx - 4
            case _:
                raise ValueError("Invalid position!")

        # Insert target int at position
        old_int = int(new_seq[idx])
        new_seq[idx] = target_int
        new_seq = np.append(new_seq, old_int)

        # Lucky, object id is already correct
        if object_ids[0] == new_target:
            pass
        # Target was a distractor, lets swap
        elif np.any(object_ids == new_target):
            idx = np.where(object_ids == new_target)[0][0]
            object_ids[idx] = object_ids[0]
            object_ids[0] = new_target
        # Add target as target
        else:
            object_ids[0] = new_target

        return object_ids, new_seq
