"""Evaluate the models to test their understanding of different lengths of sequences."""
import glob
import os
import pickle
import time
from typing import Tuple

import torch
from absl import app, flags
from joblib import Parallel, delayed
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

# Generic flags
FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "The run id to query.")
flags.DEFINE_integer(
    "n_jobs", None, "Number of jobs to run. Defaults to os.cpu_count()."
)
flags.DEFINE_string(
    "accelerator",
    "cpu",
    "Which accelerator to use for inference. Defaults to cpu.",
)


def main(unused_argv):
    # Unused, handled by absl
    del unused_argv

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

    if not os.path.exists(os.path.join(log_dir, "lightning_pt_eval")):
        os.makedirs(os.path.join(log_dir, "lightning_pt_eval"))
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

    seq_deltas = [0, 5, -5, 10, -10, 20, -20, 40, -40]

    if FLAGS.run_id is None:
        all_ids = [
            run_id.split("-")[1]
            for run_id in os.listdir(os.path.join(log_dir, "checkpoints"))
            if run_id.split("-")[2] in agents_to_query
        ]
        run_ids_to_query = list(set(all_ids))
    else:
        run_ids_to_query = [FLAGS.run_id]

    dataset_size = 20000

    # Number of trials per agent
    num_trials = 10

    # Run all evaluations in parallel
    results = Parallel(
        n_jobs=FLAGS.n_jobs if FLAGS.n_jobs is not None else os.cpu_count(), verbose=10
    )(
        delayed(eval_flow)(
            run_id=run_id,
            agents_to_query=agents_to_query,
            num_trials=num_trials,
            dataset_size=dataset_size,
            log_dir=log_dir,
            seq_deltas=seq_deltas,
        )
        for run_id in run_ids_to_query
    )

    data_dict = {}
    for res in results:
        data_dict[res[1]] = res[0]

    with open(
        os.path.join(log_dir, "lightning_pt_eval", "eval_data.pickle"), "wb"
    ) as f:
        pickle.dump(
            data_dict,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def eval_flow(
    run_id,
    agents_to_query,
    num_trials,
    dataset_size,
    log_dir,
    seq_deltas,
    device="cpu",
) -> Tuple[dict, str]:
    all_averages = {}
    for agent in agents_to_query:
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
        model.to(device=device)

        agent_averages = {}

        for seq_delta in seq_deltas:
            new_seq_len = model.hparams["seq_length"] + seq_delta

            # In this case, the senders window would become impossible
            if new_seq_len < 5:
                continue

            test_dataset = ProgressiveDataset(
                dataset_size=dataset_size,
                num_points=new_seq_len,
            )

            seq_averages = []
            for trial in range(num_trials):
                trial_average = []
                print(
                    f"Evaluating test accuracy for {agent},"
                    f" run {run_id},"
                    f" sequence length difference {seq_delta}."
                )

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

                for batch in next(iter(loader)):
                    full_sequence, cut_input, tds, target_labels = batch

                    full_sequence = full_sequence.to(device=model.device)
                    cut_input = cut_input.to(device=model.device)
                    tds = tds.to(device=model.device)
                    target_labels = target_labels.to(device=model.device)
                    with torch.no_grad():
                        guess, message = model.forward(
                            (full_sequence, cut_input, tds, target_labels)
                        )
                    accuracy = (guess == target_labels.squeeze()).float().mean()
                    trial_average.append(accuracy.item())

                trial_average = sum(trial_average) / len(trial_average)
                seq_averages.append(trial_average)

            agent_averages[seq_delta] = seq_averages
        all_averages[agent] = agent_averages

    return all_averages, run_id


if __name__ == "__main__":
    app.run(main)
