"""Query the models to test their understanding of extracted messages."""
import glob
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from absl import app, flags
from joblib import Parallel, delayed
from numpyencoder import NumpyEncoder
from torch.utils.data import DataLoader

from tpg.dataset import ProgressiveDataset
from tpg.models import (
    BaseGRUNetwork,
    BaseHybridNetwork,
    BaseLSTMNetwork,
    TemporalAttentionGRUNetwork,
    TemporalAttentionHybridNetwork,
    TemporalAttentionLSTMNetwork,
    TemporalGRUNetwork,
    TemporalHybridNetwork,
    TemporalLSTMNetwork,
)
from tpg.utils import LanguageMapper, default_to_regular, query_agent

# Generic flags
FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "The run id to query.")
flags.DEFINE_integer(
    "n_jobs", None, "Number of jobs to run. Defaults to os.cpu_count()."
)


def main(argv):
    """
    Query the models.

    Parameters
    ----------
    argv:
        Unused, as this is handled by absl
    """
    del argv  # Unused.

    # We want the logs to be contained in the main folder
    # Unless told otherwise
    full_path = os.path.realpath(__file__)
    path = os.path.split(os.path.split(full_path)[0])[0]

    # No support for custom log_dirs for now
    log_dir = os.path.join(path, "logs")

    # Check whether the specified paths exist or not and create them
    # Sleep is to make sure change is committed
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_chat")):
        os.makedirs(os.path.join(log_dir, "lightning_chat"))
        time.sleep(1)

    agents_to_query = {
        "BaseGRU": BaseGRUNetwork,
        # "TemporalGRU": TemporalGRUNetwork,
        # "TemporalAttentionGRU": TemporalAttentionGRUNetwork,
        # "BaseLSTM": BaseLSTMNetwork,
        # "TemporalLSTM": TemporalLSTMNetwork,
        # "TemporalAttentionLSTM": TemporalAttentionLSTMNetwork,
        # "BaseHybrid": BaseHybridNetwork,
        # "TemporalHybrid": TemporalHybridNetwork,
        # "TemporalAttentionHybrid": TemporalAttentionHybridNetwork,
    }

    top_ns = [1, 2, 3, 5, 10, 15]
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with open("./analysis/dictionary_nc.pickle", "rb") as handle:
        non_compo_full_translation_dict = pickle.load(handle)

    with open("./analysis/dictionary_c.pickle", "rb") as handle:
        compo_full_translation_dict = pickle.load(handle)

    agent_accuracy_full = {
        x: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for x in agents_to_query
    }
    agent_dict_lens_full = {
        x: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for x in agents_to_query
    }

    for top_n in top_ns:
        for confidence in confidences:
            run_ids_analysed = set()
            current_non_compo_dict = non_compo_full_translation_dict[
                f"topn_{top_n}-confidence_{confidence}"
            ]
            current_compo_dict = compo_full_translation_dict[
                f"topn_{top_n}-confidence_{confidence}"
            ]
            # We clamp the positional dict
            # Since by empirical tests on synthetic languages the highest NPMI is at most ~0.3
            # and the highest accuracy is at 0.3
            current_compo_dict_positionals = compo_full_translation_dict[
                f"topn_{top_n}-confidence_{0.3}"
            ]

            # We only need one dict for the matches, since the matches analysed will be the same
            for match in current_non_compo_dict:
                run_ids_analysed.add(current_non_compo_dict[match]["run_id"])

            # In case only a specific match is requested
            if FLAGS.run_id is None:
                run_ids_to_query = run_ids_analysed
            else:
                run_ids_to_query = [FLAGS.run_id]

            # Use a sizeable dataset for good quality of evaluation
            dataset_size = 20000

            # Interrogation flow.
            # Number of trials per agent, to evaluate difference between random and dict
            num_trials = 10
            # Run all evaluations in parallel
            results = Parallel(
                n_jobs=FLAGS.n_jobs if FLAGS.n_jobs is not None else os.cpu_count(),
                verbose=10,
            )(
                delayed(query_flow)(
                    run_id=run_id,
                    top_n=top_n,
                    confidence=confidence,
                    agents_to_query=agents_to_query,
                    non_compo_dict=current_non_compo_dict,
                    compo_dict=current_compo_dict,
                    compo_dict_positionals=current_compo_dict_positionals,
                    num_trials=num_trials,
                    dataset_size=dataset_size,
                    log_dir=log_dir,
                )
                for run_id in run_ids_to_query
            )

            # Combine the parallel results into one dict
            for res in results:
                agent_dict_lens, agent_accuracy = res
                for agent in agent_dict_lens:
                    for translation_type in agent_dict_lens[agent]:
                        for seq_neg in agent_dict_lens[agent][translation_type]:
                            for length in agent_dict_lens[agent][translation_type][
                                seq_neg
                            ]:
                                agent_dict_lens_full[agent][
                                    f"topn_{top_n}-confidence_{confidence}"
                                ][translation_type][seq_neg].append(length)
                            for acc in agent_accuracy[agent][translation_type][seq_neg]:
                                agent_accuracy_full[agent][
                                    f"topn_{top_n}-confidence_{confidence}"
                                ][translation_type][seq_neg].append(acc)

            for agent in agents_to_query:
                for translation_type in agent_accuracy_full[agent][
                    f"topn_{top_n}-confidence_{confidence}"
                ]:
                    for seq_neg in agent_accuracy_full[agent][
                        f"topn_{top_n}-confidence_{confidence}"
                    ][translation_type]:
                        mean_calc = np.array(
                            agent_accuracy_full[agent][
                                f"topn_{top_n}-confidence_{confidence}"
                            ][translation_type][seq_neg]
                        ).mean()
                        std_calc = np.array(
                            agent_accuracy_full[agent][
                                f"topn_{top_n}-confidence_{confidence}"
                            ][translation_type][seq_neg]
                        ).std()
                        print(
                            f"Accuracy for {agent} with translation {translation_type}, "
                            f"for top_n {top_n} and confidence {confidence} and seq_neg {seq_neg}: "
                            f"mean {mean_calc},"
                            f" std {std_calc}."
                        )

    print("Creating the accuracy dict for later analysis")

    agent_accuracy_full = default_to_regular(agent_accuracy_full)
    agent_dict_lens_full = default_to_regular(agent_dict_lens_full)

    with open(
        os.path.join(log_dir, "lightning_chat", "agent_accuracy_full.pickle"), "wb"
    ) as handle:
        pickle.dump(
            agent_accuracy_full,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open(
        os.path.join(log_dir, "lightning_chat", "agent_dict_lens_full.pickle"), "wb"
    ) as handle:
        pickle.dump(
            agent_dict_lens_full,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def query_flow(
    run_id,
    top_n,
    confidence,
    agents_to_query,
    non_compo_dict,
    compo_dict,
    compo_dict_positionals,
    num_trials,
    dataset_size,
    log_dir,
):
    """

    Help run the querying in parallel.

    Parameters
    ----------
    run_id: str
        Run id of the run to query.
    top_n: int
        Top_n integers to consider.
    confidence: float
        Confidence cut-off for NPMI measure.
    agents_to_query: dict
        Dictionary of which agents to query mapping to their class
    non_compo_dict: dict
        Dictionary for non-compositional messages
    compo_dict: dict
        Dictionary for compositional messages.
    compo_dict_positionals: dict
        Dictionary for positional references.
    num_trials: int
        Number of trials per agent to run
    dataset_size: int
        Size of the dataset for evaluation.
    log_dir: os.Path
        Path to the log directory.

    Returns
    -------
    agent_dict_lens: dict
    agent_accuracy: dict

    """
    print(f"Starting querying of run {run_id} for {top_n=} and {confidence=}.")
    agent_dict_lens = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    agent_accuracy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    shorter_seq = [0, 5, 10, 20, 40]

    for agent in agents_to_query:
        agent_compo_translation_dict = pd.DataFrame(compo_dict).T
        agent_compo_positionals_translation_dict = pd.DataFrame(
            compo_dict_positionals
        ).T
        agent_non_compo_translation_dict = pd.DataFrame(non_compo_dict).T

        positional_translation_dict = agent_non_compo_translation_dict[
            (agent_non_compo_translation_dict["arch"] == agent)
            & (agent_non_compo_translation_dict["run_id"] == run_id)
        ]["positional_messages"]
        positional_translation_dict = (
            positional_translation_dict.iloc[0]
            if positional_translation_dict.any()
            else {}
        )
        non_compo_translation_dict = agent_non_compo_translation_dict[
            (agent_non_compo_translation_dict["arch"] == agent)
            & (agent_non_compo_translation_dict["run_id"] == run_id)
        ]["other_messages"]
        non_compo_translation_dict = (
            non_compo_translation_dict.iloc[0]
            if non_compo_translation_dict.any()
            else {}
        )

        compo_translation_dict_integer = agent_compo_translation_dict[
            (agent_compo_translation_dict["arch"] == agent)
            & (agent_compo_translation_dict["run_id"] == run_id)
        ]["integer_ngrams"]
        compo_translation_dict_integer = (
            compo_translation_dict_integer.iloc[0]
            if compo_translation_dict_integer.any()
            else {}
        )

        compo_translation_dict_positional = agent_compo_positionals_translation_dict[
            (agent_compo_positionals_translation_dict["arch"] == agent)
            & (agent_compo_positionals_translation_dict["run_id"] == run_id)
        ]["positional_ngrams"]
        compo_translation_dict_positional = (
            compo_translation_dict_positional.iloc[0]
            if compo_translation_dict_positional.any()
            else {}
        )

        # Remove empty entries
        positional_translation_dict = {
            k: v for k, v in positional_translation_dict.items() if v
        }
        non_compo_translation_dict = {
            k: v for k, v in non_compo_translation_dict.items() if v
        }
        compo_translation_dict_integer = {
            k: v for k, v in compo_translation_dict_integer.items() if v
        }
        compo_translation_dict_positional = {
            k: v for k, v in compo_translation_dict_positional.items() if v
        }

        # Compositional integer translation dictionary needs some pruning
        # Since parts of it are accounted for before
        # i.e., the length 3 ngrams are in non-compositional
        for pos in compo_translation_dict_integer:
            compo_translation_dict_integer[pos] = {
                integer: [
                    msg
                    for msg in compo_translation_dict_integer[pos][integer]
                    if len(msg) < 3
                ]  # Ignore length 3 n-grams, as they are accounted for in the non-compositional analysis
                for integer in compo_translation_dict_integer[pos]
            }

        # Remove empty positions
        for pos in compo_translation_dict_integer:
            int_keys = list(compo_translation_dict_integer[pos].keys())
            for integer in int_keys:
                if not compo_translation_dict_integer[pos][integer]:
                    del compo_translation_dict_integer[pos][integer]
        pos_keys = list(compo_translation_dict_integer.keys())
        for pos in pos_keys:
            if not compo_translation_dict_integer[pos]:
                del compo_translation_dict_integer[pos]

        # Remove empty positions
        for pos in compo_translation_dict_positional:
            ref_pos_keys = list(compo_translation_dict_positional[pos].keys())
            for ref_pos in ref_pos_keys:
                if not compo_translation_dict_positional[pos][ref_pos]:
                    del compo_translation_dict_positional[pos][ref_pos]
        pos_keys = list(compo_translation_dict_positional.keys())
        for pos in pos_keys:
            if not compo_translation_dict_positional[pos]:
                del compo_translation_dict_positional[pos]

        combined_dict_len = (
            len(positional_translation_dict)
            + len(non_compo_translation_dict)
            + len(compo_translation_dict_integer)
            + len(compo_translation_dict_positional)
        )

        # No point for this agent pair
        if combined_dict_len == 0:
            continue

        for translation_type in [
            # None,
            "Positional",
            "NonCompositional",
            "Compositional-P",  # With positionals
            "Compositional-NP",  # Without positionals
        ]:  # Could do None, but its always 20%
            positional = False
            non_compo = False
            compo = False
            c_pos = False
            no_ints = False

            if translation_type == "Positional":
                dict_len = sum(len(v) for v in positional_translation_dict.values())
                positional = True
                special_to_gen = {
                    "pos": [k for k, v in positional_translation_dict.items()],
                }
            elif translation_type == "NonCompositional":
                dict_len = sum(len(v) for v in non_compo_translation_dict.values())
                non_compo = True
                special_to_gen = {
                    "nc": non_compo_translation_dict,
                }
            elif translation_type == "Compositional-P":
                dict_len = sum(len(v) for v in compo_translation_dict_integer.values())
                # If there is not ints there is nothing to be described
                if dict_len == 0:
                    no_ints = True
                dict_len += sum(
                    len(v) for v in compo_translation_dict_positional.values()
                )
                compo = True
                c_pos = True
                special_to_gen = {
                    "c-int": compo_translation_dict_integer,
                    "c-pos": compo_translation_dict_positional,
                }
            elif translation_type == "Compositional-NP":
                dict_len = sum(len(v) for v in compo_translation_dict_integer.values())
                compo = True
                c_pos = False
                special_to_gen = {
                    "c-int": compo_translation_dict_integer,
                }
            elif translation_type is None:
                dict_len = -1
                special_to_gen = {}
            else:
                raise ValueError(f"Unknown translation type: {translation_type}")

            if dict_len == 0 or no_ints:
                print(
                    f"Skipping {agent}, run {run_id}, as there are no messages for {translation_type}"
                )
                continue

            agent_dict_lens[agent][translation_type]["None"].append(dict_len)

            for seq_neg in shorter_seq:
                mistakes_dict = defaultdict(list)
                correct_dict = defaultdict(list)
                print(
                    f"Evaluating translation dictionary for {agent},"
                    f" run {run_id},"
                    f" sequence len -{seq_neg},"
                    f" and using translation {translation_type}."
                )
                print(f"Size of translation dictionary is {dict_len}.")
                all_checkpoints = glob.glob(
                    os.path.join(
                        log_dir,
                        "checkpoints",
                        f"run-{run_id}-{agent}",
                        "*.ckpt",
                    )
                )
                model = agents_to_query[agent].load_from_checkpoint(all_checkpoints[0])
                model.eval()

                lang = LanguageMapper(
                    message_length=model.hparams["max_length"],
                    vocab_size=model.hparams["vocab_size"],
                    positional_message_dict=positional_translation_dict
                    if positional
                    else {},
                    non_compo_message_dict=non_compo_translation_dict
                    if non_compo
                    else {},
                    compo_message_dict_int=compo_translation_dict_integer
                    if compo
                    else {},
                    compo_message_dict_pos=compo_translation_dict_positional
                    if c_pos
                    else {},
                )
                for trial_no in range(num_trials):
                    test_dataset = ProgressiveDataset(
                        # seed=42,
                        dataset_size=dataset_size,
                        num_points=model.hparams["seq_length"] - seq_neg,
                        num_distractors=model.hparams["num_distractors"],
                        repeat_chance=0.25,
                        sequence_window=True,
                        sequence_window_size=model.hparams["seq_window"],
                        use_random=True,
                        generate_special=special_to_gen
                        if translation_type is not None
                        else None,
                    )

                    # https://github.com/joblib/joblib/issues/1104
                    # https://github.com/pytorch/pytorch/issues/44687
                    loader = (
                        DataLoader(
                            test_dataset,
                            batch_size=2048,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            # This is needed for DataLoader to work with joblib Parallel
                            multiprocessing_context="fork",
                        ),
                    )

                    # The code below comes directly from the relevant agent classes
                    true_values = 0
                    for batch in next(iter(loader)):
                        full_sequence, cut_input, tds, target_labels = batch

                        (_, meanings_c), message = lang.map_obs_message(cut_input)
                        message = torch.tensor(message, device=model.device)
                        full_sequence = full_sequence.to(device=model.device)
                        tds = tds.to(device=model.device)
                        target_labels = target_labels.to(device=model.device)

                        guess, message = query_agent(
                            full_sequence=full_sequence,
                            tds=tds,
                            model=model,
                            message=message,
                            agent=agent,
                        )

                        # Get labels of correct and incorrect predictions to see what went wrong
                        if translation_type is not None:
                            bool_wrong_labels = (
                                (guess.argmax(dim=1) != target_labels.squeeze())
                                .detach()
                                .cpu()
                            )
                            bool_correct_labels = (
                                (guess.argmax(dim=1) == target_labels.squeeze())
                                .detach()
                                .cpu()
                            )
                            for dict_type, label_type in zip(
                                [mistakes_dict, correct_dict],
                                [bool_wrong_labels, bool_correct_labels],
                            ):
                                dict_type["inputs"].append(cut_input[label_type])
                                dict_type["targets"].append(
                                    torch.gather(
                                        tds, 1, target_labels.to(dtype=torch.int64)
                                    )[label_type]
                                )
                                dict_type["guesses"].append(
                                    torch.gather(
                                        tds,
                                        1,
                                        guess.argmax(dim=1)
                                        .unsqueeze(-1)
                                        .to(dtype=torch.int64),
                                    )[label_type]
                                )
                                dict_type["messages"].append(
                                    message[label_type].argmax(dim=2)
                                )
                                dict_type["meanings"].append(
                                    lang.map_message_meaning(
                                        message[label_type].argmax(dim=2)
                                    )
                                    if translation_type
                                    not in [
                                        "Compositional-P",
                                        "Compositional-NP",
                                    ]
                                    else np.array(meanings_c)[label_type]
                                )

                        true_values += (
                            torch.sum(guess.argmax(dim=1) == target_labels.squeeze())
                            .detach()
                            .cpu()
                        )

                    agent_accuracy[agent][translation_type][seq_neg].append(
                        float(true_values / dataset_size)
                    )

                if translation_type is not None:
                    for dict_type, name in zip(
                        [mistakes_dict, correct_dict], ["mistakes", "correct"]
                    ):
                        for k in dict_type:
                            if k != "meanings":
                                dict_type[k] = torch.vstack(dict_type[k]).cpu().numpy()
                        dict_type = dict(dict_type)
                        with open(
                            os.path.join(
                                log_dir,
                                "lightning_chat",
                                f"{agent}-{run_id}-translation_{translation_type}"
                                f"-topn_{top_n}-confidence_{confidence}-seqneg_{seq_neg}-{name}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(
                                dict_type,
                                f,
                                cls=NumpyEncoder,
                            )

    return agent_dict_lens, agent_accuracy


if __name__ == "__main__":
    app.run(main)
