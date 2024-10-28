"""Module will run a very long test of the dataset generation."""
from tpg.dataset import ProgressiveDataset

if __name__ == "__main__":
    import pickle

    import pandas as pd

    # The below needs these files to be created using synthetic_languages.ipynb
    with open("../analysis/dictionary_c_debug.pickle", "rb") as handle:
        c_dicts_debug = pickle.load(handle)
    with open("../analysis/dictionary_nc_debug.pickle", "rb") as handle:
        nc_dicts_debug = pickle.load(handle)

    top_ns = [1, 2, 3, 5, 10, 15]
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    num_trials = 10

    for top_n in top_ns:
        for confidence in confidences:
            run_ids_analysed = set()
            current_non_compo_dict = nc_dicts_debug[
                f"topn_{top_n}-confidence_{confidence}"
            ]
            current_compo_dict = c_dicts_debug[f"topn_{top_n}-confidence_{confidence}"]

            for match in current_non_compo_dict:
                run_ids_analysed.add(current_non_compo_dict[match]["run_id"])

            # In case only a specific match is requested
            run_ids_to_query = run_ids_analysed
            for agent in [
                "synthetic_nc_spec",
                "synthetic_pos",
                "synthetic_posinv",
                "synthetic_pos_spec",
            ]:
                for run_id in run_ids_to_query:
                    agent_compo_translation_dict = pd.DataFrame(current_compo_dict).T
                    agent_non_compo_translation_dict = pd.DataFrame(
                        current_non_compo_dict
                    ).T

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

                    compo_translation_dict_positional = agent_compo_translation_dict[
                        (agent_compo_translation_dict["arch"] == agent)
                        & (agent_compo_translation_dict["run_id"] == run_id)
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
                        ref_pos_keys = list(
                            compo_translation_dict_positional[pos].keys()
                        )
                        for ref_pos in ref_pos_keys:
                            if not compo_translation_dict_positional[pos][ref_pos]:
                                del compo_translation_dict_positional[pos][ref_pos]
                    pos_keys = list(compo_translation_dict_positional.keys())
                    for pos in pos_keys:
                        if not compo_translation_dict_positional[pos]:
                            del compo_translation_dict_positional[pos]

                    for translation_type in [
                        # None,
                        # "Positional",
                        # "NonCompositional",
                        "Compositional-P",  # With positionals
                        "Compositional-NP",  # Without positionals
                    ]:  # Could do None, but its always 20%
                        positional = False
                        non_compo = False
                        compo = False
                        c_pos = False

                        if translation_type == "Positional":
                            dict_len = sum(
                                len(v) for v in positional_translation_dict.values()
                            )
                            positional = True
                            special_to_gen = {
                                "pos": [
                                    k for k, v in positional_translation_dict.items()
                                ],
                            }
                        elif translation_type == "NonCompositional":
                            dict_len = sum(
                                len(v) for v in non_compo_translation_dict.values()
                            )
                            non_compo = True
                            special_to_gen = {
                                "nc": non_compo_translation_dict,
                            }
                        elif translation_type == "Compositional-P":
                            dict_len = sum(
                                len(v) for v in compo_translation_dict_integer.values()
                            )
                            dict_len += sum(
                                len(v)
                                for v in compo_translation_dict_positional.values()
                            )
                            compo = True
                            c_pos = True
                            special_to_gen = {
                                "c-int": compo_translation_dict_integer,
                                "c-pos": compo_translation_dict_positional,
                            }
                        elif translation_type == "Compositional-NP":
                            dict_len = sum(
                                len(v) for v in compo_translation_dict_integer.values()
                            )
                            compo = True
                            c_pos = False
                            special_to_gen = {
                                "c-int": compo_translation_dict_integer,
                            }
                        elif translation_type is None:
                            dict_len = -1
                            special_to_gen = {}
                        else:
                            raise ValueError(
                                f"Unknown translation type: {translation_type}"
                            )

                        if dict_len == 0:
                            print(
                                f"Skipping {agent}, run {run_id}, as there are no messages for {translation_type}"
                            )
                            continue

                        for trial_no in range(num_trials):
                            test_dataset = ProgressiveDataset(
                                # seed=42,
                                dataset_size=20000,
                                num_points=60,
                                num_distractors=4,
                                repeat_chance=0.25,
                                sequence_window=True,
                                sequence_window_size=2,
                                use_random=True,
                                generate_special=special_to_gen
                                if translation_type is not None
                                else None,
                            )
                        print(f"Finished {translation_type} for agent {agent}")
