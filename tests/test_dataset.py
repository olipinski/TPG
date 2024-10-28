"""Dataset testing logic."""
import numpy as np
import pytest

from tpg.dataset import ProgressiveDataset


@pytest.mark.parametrize("seed", [42, 8])
@pytest.mark.parametrize("dataset_size", [10, 1000])
@pytest.mark.parametrize("num_points", [20, 100])
@pytest.mark.parametrize("num_distractors", [1, 4])
@pytest.mark.parametrize("repeat_chance", [0, 0.5])
@pytest.mark.parametrize("sequence_window", (True, False))
@pytest.mark.parametrize("sequence_window_size", [1, 2])
@pytest.mark.parametrize("use_random", (True, False))
@pytest.mark.parametrize(
    "generate_special",
    [
        None,
        {"pos": ["begin"]},
        {"pos": ["begin", "begin+1", "end-1", "end"]},
        {"nc": {"l1": {0: [np.array([20, 9, 9])]}, "l2": {0: [np.array([7, 16, 7])]}}},
        {
            "pos": ["begin", "begin+1", "end-1", "end"],
        },
        {
            "c-int": {
                "inv_npmi": {
                    0: np.array([20, 9, 9]),
                    1: np.array([7, 16, 7]),
                    20: np.array([7, 16, 7]),
                },
                "npmi_pos_0": {
                    2: np.array([20, 9, 9]),
                    5: np.array([7, 16, 7]),
                    8: np.array([7, 16, 7]),
                },
                "npmi_pos_1": {
                    3: np.array([20, 9, 9]),
                    9: np.array([7, 16, 7]),
                    10: np.array([7, 16, 7]),
                },
                "npmi_pos_2": {
                    21: np.array([20, 9, 9]),
                    15: np.array([7, 16, 7]),
                    40: np.array([7, 16, 7]),
                },
            },
        },
    ],
)
def test_progressive_dataset(
    seed,
    dataset_size,
    num_points,
    num_distractors,
    repeat_chance,
    sequence_window,
    sequence_window_size,
    use_random,
    generate_special,
):
    """
    Test the dataset generation, and the correctness of the dataset labelling.

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
    # Special cases for generate special, these are **not** coded with generality in mind
    # So they may only work for the specific values in the paper
    test_dataset = ProgressiveDataset(
        seed=seed,
        dataset_size=dataset_size,
        num_points=num_points if not generate_special else 60,
        num_distractors=num_distractors,
        repeat_chance=repeat_chance,
        sequence_window=sequence_window if not generate_special else True,
        sequence_window_size=sequence_window_size if not generate_special else 2,
        use_random=use_random if not generate_special else True,
        generate_special=generate_special,
    )

    assert test_dataset

    for sequence, cut_inputs, tds, label in test_dataset:
        # Check all values are unique
        assert len(set(tds)) == len(tds)
        # Check if the target is actually the missing number
        # First find the supposed target number in targets and distractors
        target = tds[label]
        # Put it into the smaller sequence
        cut_inputs[np.where(cut_inputs == -1)[0][0]] = target
        # Now check if this sequence is still part of the larger sequence
        # We use simple string matching
        a = [
            i
            for i in range(0, len(sequence))
            if list(sequence[i : i + len(cut_inputs)]) == list(cut_inputs)
        ]
        # This subsequence should only appear once
        assert len(a) == 1
