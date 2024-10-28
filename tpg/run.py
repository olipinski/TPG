"""A simple neural network to check for spatial deixis."""
import json
import os
import platform
import time

import lightning as L
import numpy as np
import shortuuid
import torch
import torch.nn.functional as F
import wandb
from absl import app, flags
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
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

# Generic flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "seed", None, "The seed to use. If none is provided a random seed is generated."
)
flags.DEFINE_integer("max_epochs", 200, "Maximum number of epochs to train for.")

flags.DEFINE_string("wandb_group", None, "Name of the WandB group to put the run into.")
flags.DEFINE_bool("wandb_offline", False, "Whether to run WandB in offline mode.")

# Dataset Flags
flags.DEFINE_integer(
    "dataset_size", 262144, "Size of the dataset to generate for training."
)

flags.DEFINE_integer(
    "num_distractors", 4, "Number of distractor objects to show to the receiver"
)

flags.DEFINE_integer("num_points", 60, "Number of points to")

flags.DEFINE_bool(
    "sequence_window",
    True,
    "Whether to pass a window or full view of the sequence with missing items.",
)

flags.DEFINE_integer(
    "sequence_window_size",
    2,
    "Size of half of sequence window visible to the sender. "
    "This size refers to the extent on each side of the target number, e.g., [0,1,...,-1,...,2n]",
)

flags.DEFINE_float(
    "repeat_chance",
    0.0,
    "Chance that a target will repeat.",
)

flags.DEFINE_bool(
    "use_random",
    True,
    "Whether to use a random dataset sequence, or x=y (1,2,3,4,5,6). Defaults to using a random sequence",
)

# Communication flags
flags.DEFINE_integer(
    "message_length",
    3,
    "The maximum length of the message that the sender agent can send",
)

flags.DEFINE_integer(
    "vocab_size", 26, "The size of the available vocabulary to the sender"
)

# Agent flags
flags.DEFINE_float(
    "length_penalty",
    0.0,
    "Factor by which length penalty will be multiplied. "
    "Setting to 0 disables the length penalty. Higher values (>0.001) may lead to unstable training.",
)

flags.DEFINE_integer(
    "sender_hidden", 64, "Hidden size of the sender LSTMs and attention."
)

flags.DEFINE_integer(
    "receiver_hidden", 64, "Hidden size of the receiver LSTMs and attention."
)


def main(argv):
    """
    Run the training and evaluation.

    Parameters
    ----------
    argv:
        Unused, as this is handled by absl
    """
    del argv  # Unused.

    # Lightning sets it as dry run instead of offline
    if FLAGS.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

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

    if not os.path.exists(os.path.join(log_dir, "lightning_tensorboard")):
        os.makedirs(os.path.join(log_dir, "lightning_tensorboard"))
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_wandb")):
        os.makedirs(os.path.join(log_dir, "lightning_wandb"))
        time.sleep(1)

    if not os.path.exists(os.path.join(log_dir, "lightning_interactions")):
        os.makedirs(os.path.join(log_dir, "lightning_interactions"))
        time.sleep(1)

    run_uuid = shortuuid.uuid()[:8]

    # Create training dataset
    train_ds = ProgressiveDataset(
        seed=FLAGS.seed,
        dataset_size=FLAGS.dataset_size,
        num_points=FLAGS.num_points,
        num_distractors=FLAGS.num_distractors,
        repeat_chance=FLAGS.repeat_chance,
        sequence_window=FLAGS.sequence_window,
        sequence_window_size=FLAGS.sequence_window_size,
        use_random=FLAGS.use_random,
    )

    # Create validation dataset
    val_ds = ProgressiveDataset(
        seed=FLAGS.seed,
        dataset_size=FLAGS.dataset_size,
        num_points=FLAGS.num_points,
        num_distractors=FLAGS.num_distractors,
        repeat_chance=FLAGS.repeat_chance,
        sequence_window=FLAGS.sequence_window,
        sequence_window_size=FLAGS.sequence_window_size,
        use_random=FLAGS.use_random,
    )

    trainer_list = []

    agents_to_train = {
        "BaseGRU_OHV": BaseGRUNetwork(
            num_distractors=FLAGS.num_distractors,
            seq_length=FLAGS.num_points,
            seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
            max_length=FLAGS.message_length,
            vocab_size=FLAGS.vocab_size,
            length_penalty=FLAGS.length_penalty,
            sender_hidden=FLAGS.sender_hidden,
            receiver_hidden=FLAGS.receiver_hidden,
            one_hot=True,
        ),
        "BaseGRU": BaseGRUNetwork(
            num_distractors=FLAGS.num_distractors,
            seq_length=FLAGS.num_points,
            seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
            max_length=FLAGS.message_length,
            vocab_size=FLAGS.vocab_size,
            length_penalty=FLAGS.length_penalty,
            sender_hidden=FLAGS.sender_hidden,
            receiver_hidden=FLAGS.receiver_hidden,
            one_hot=False,
        ),
        # "TemporalGRU": TemporalGRUNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "TemporalAttentionGRU": TemporalAttentionGRUNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "BaseLSTM": BaseLSTMNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "TemporalLSTM": TemporalLSTMNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "TemporalAttentionLSTM": TemporalAttentionLSTMNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "BaseHybrid": BaseHybridNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "TemporalHybrid": TemporalHybridNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
        # "TemporalAttentionHybrid": TemporalAttentionHybridNetwork(
        #     num_distractors=FLAGS.num_distractors,
        #     seq_length=FLAGS.num_points,
        #     seq_window=FLAGS.sequence_window_size if FLAGS.sequence_window else 0,
        #     max_length=FLAGS.message_length,
        #     vocab_size=FLAGS.vocab_size,
        #     length_penalty=FLAGS.length_penalty,
        #     sender_hidden=FLAGS.sender_hidden,
        #     receiver_hidden=FLAGS.receiver_hidden,
        # ),
    }

    # Check GPU capability for compile
    compile_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap[0] >= 7:
            compile_ok = True
        if platform.uname()[0] == "Windows":
            compile_ok = False

    # Speedups
    torch.set_float32_matmul_precision("high")

    run_count = 0
    max_runs = len(agents_to_train)
    print(f"Global run ID: {run_uuid}")
    for agent in agents_to_train.keys():
        print(f"Starting training run {run_count+1} out of {max_runs}")
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(log_dir, "lightning_tensorboard"),
            name=f"run-{run_uuid}-{agent}",
        )
        wandb_logger = WandbLogger(
            project="TPGv5",
            save_dir=os.path.join(log_dir, "lightning_wandb"),
            group=FLAGS.wandb_group,
            name=f"run-{run_uuid}-{agent}",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(
                log_dir,
                "checkpoints",
                f"run-{run_uuid}-{agent}",
            ),
            monitor="val_acc",
            save_top_k=3,
            mode="max",
        )
        network = agents_to_train[agent]

        # TODO this is broken in lightning 2.2
        if compile_ok:
            pass
        #    network = torch.compile(network)

        wandb_logger.experiment.config.update(
            {
                "architecture": agent,
                "dataset_size": FLAGS.dataset_size,
                "num_distractors": FLAGS.num_distractors,
                "message_length": FLAGS.message_length,
                "vocab_size": FLAGS.vocab_size,
                "repeat_chance": FLAGS.repeat_chance,
                "seed": FLAGS.seed,
                "max_epochs": FLAGS.max_epochs,
                "length_penalty": FLAGS.length_penalty,
            }
        )

        trainer = L.Trainer(
            accelerator="auto",
            max_epochs=FLAGS.max_epochs,
            logger=[tb_logger, wandb_logger],
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=5,
            strategy="auto",
        )

        trainer.fit(
            network,
            DataLoader(
                train_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
            ),
            DataLoader(
                val_ds,
                batch_size=2048,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
            ),
        )
        # Otherwise WandB logger will reuse the run which we don't want.
        wandb.finish()
        trainer_list.append(
            (
                trainer,
                f"run-{run_uuid}-{agent}",
            )
        )
        run_count += 1

    # Test flow, including the dumping of the test dictionaries for later analysis.
    run_count = 0
    for trainer, trainer_str in trainer_list:
        print(f"Starting evaluation run {run_count+1} out of {max_runs}")
        print(f"Evaluation predictions for {trainer_str}")

        test_dataset_size = FLAGS.dataset_size

        # Very hacky way to clear the exchange dict...
        trainer.strategy._lightning_module.exchange_dict = {}
        trainer.strategy._lightning_module.exchange_count = 0

        test_dataset = ProgressiveDataset(
            seed=FLAGS.seed,
            dataset_size=test_dataset_size,
            num_points=FLAGS.num_points,
            num_distractors=FLAGS.num_distractors,
            repeat_chance=FLAGS.repeat_chance,
            sequence_window=FLAGS.sequence_window,
            sequence_window_size=FLAGS.sequence_window_size,
            use_random=FLAGS.use_random,
        )

        # Remove loggers as they break predictions
        trainer._loggers = []
        predictions_test = trainer.predict(
            dataloaders=DataLoader(
                test_dataset,
                batch_size=2048,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
            ),
            ckpt_path="last",
        )

        # Save interactions
        # Slightly hacky but it works.
        with open(
            os.path.join(
                log_dir,
                "lightning_interactions",
                f"{trainer_str}-interactions.json",
            ),
            "w",
        ) as f:
            json.dump(
                trainer.strategy._lightning_module.exchange_dict,
                f,
                cls=NumpyEncoder,
            )

        predictions = torch.cat(
            [prediction[0] for prediction in predictions_test]
        ).squeeze()

        true_values = 0
        labels_pred = []
        for x in range(test_dataset_size):
            labels_pred.append(test_dataset[x][3][0])
            if predictions[x] == test_dataset[x][3][0]:
                true_values += 1

        labels_pred = np.array(labels_pred)

        loss_pred = F.cross_entropy(
            predictions.float(), torch.tensor(labels_pred).float().squeeze()
        )

        print(f"Test predictions n_correct: {true_values} out of {test_dataset_size}")
        print(f"Test predictions loss: {loss_pred}")

        run_count += 1


if __name__ == "__main__":
    app.run(main)
