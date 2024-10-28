# Temporal Progression Games

## About

Code accompanying our paper "Speaking Your Language: Spatial Relationships in
Interpretable Emergent Communication", presented at NeurIPS 2024.

This code is based on the work of https://github.com/olipinski/TRG.

## Installation

The installation procedure has only been tested on Linux. It should however work
just fine on Windows and, with minor changes, on MacOS.

### Conda

```shell
conda env create -f environment.yml
```

## Running

```shell
python -m tpg.run [OPTIONS]
```

Possible command line options are available with

```shell
python -m tpg.run --help
```

## Reproducing the paper results

To reproduce the results from the paper, please run the following command x16.

```shell
python -m tpg.run \
        --max_epochs 1000 \
        --wandb_group e1000-200k-60p-4d-2s \
        --wandb_offline \
        --dataset_size 200000 \
        --num_distractors 4 \
        --num_points 60 \
        --sequence_window \
        --sequence_window_size 2 \
        --use_random \
        --message_length 3 \
        --vocab_size 26 \
        --length_penalty 0.0 \
        --sender_hidden 64 \
        --receiver_hidden 64
```

## Code Structure and Documentation

```
- tpg # Top directory
    - analysis
        - data # Data to analyse
        - analysis.ipynb # Analyse the data as generated by run.py, and query_model.py
        - synthethic_languages.ipynb # Create and analyse synthetic languages using the NPMI method
    - tpg # Main module
        - models # Contains all the different models available in this work. For the paper we use only the BaseGRU model.
        - utils
            - gumbel_softmax.py # Adapted from EGG for GS Discretisation
            - causal_mask.py # Causal masking for attention layers
            - positional_encoding.py # Positional encoding module for attention
            - dict_utils.py # Dictionary manipulation
            - langauge_mapper.py # Allows for mapping to/from observations and emergent language, using the NPMI dicts
            - npmi.py # Methods used for the NPMI calculations
            - query_utils.py # Helper methods for agent querying
        - dataset.py # Dataset generation
        - query_model.py # Querying of the receiver agent using inferred messages
        - eval_model.py # Evaluate both agents
        - run.py # Main entry point
```

For more details please refer to the Python files.

## Limitations

Due to probability estimation in the NPMI measures, the value of the measure is
not numerically stable, meaning the values of NPMI can sometimes go beyond the
theoretical codomain of $\[-1,1\]$. This could be improved at the cost of
computational efficiency by calculating the exact probabilities for each of the
equation constituents, for example $p(integer)$.

Only the BaseGRU model supports one-hot encoding. Based on tests the OHV model
performs worse than the scalar model, so this has not been implemented for other
models.

## Issues

- Torch Compile has some issues with Lightning logging -
  https://github.com/Lightning-AI/pytorch-lightning/issues/18835
