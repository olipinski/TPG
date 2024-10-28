"""Function for calculating the NPMI."""
import copy
import math
from collections import defaultdict

import numpy as np


def compute_non_compositional_npmi(match, top_n: int) -> dict:
    """
    Compute non-compositional NPMI.

    This will find non-compositional messages.

    Parameters
    ----------
    match: dict
        Dictionary of matches. For format please see analysis.ipynb.
    top_n: int
        Number of integer candidates to consider when computing non-compositional NPMI.

    Returns
    -------
    non_compositional_npmi_dict: dict
        Dictionary of non-compositional messages.

    """
    match_tpg_dict = match["tpg_stats"]
    total_messages = total_obs = len(match["cut_inputs"])
    non_compositional_npmi_dict = defaultdict(dict)
    for msg in match_tpg_dict:
        msg_occurrences = match_tpg_dict[msg]["total"]
        msg_prob = msg_occurrences / total_messages
        # Check for special cases first
        for special in ["begin", "begin+1", "end-1", "end"]:
            obs_occurrences = match["obs_counts"][special]
            obs_prob = obs_occurrences / total_obs
            joint_occurrences = match_tpg_dict[msg][special]
            if type(joint_occurrences) is not int:
                joint_occurrences = 0
            joint_prob = joint_occurrences / total_obs
            joint_self_inf = -np.log2(joint_prob)
            non_compositional_npmi_dict[msg][special] = (
                np.log2(joint_prob / (msg_prob * obs_prob)) / joint_self_inf
            )
        # Check for non-compositional references other than the special cases
        for pos in match_tpg_dict[msg]["obs_neighbours"]:
            pos_dict = match_tpg_dict[msg]["obs_neighbours"][pos]
            top_n_tuples = sorted(
                pos_dict.items(), key=lambda pair: pair[1], reverse=True
            )[:top_n]
            top_n_keys = [x[0] for x in top_n_tuples]
            top_n_sum = sum([x[1] for x in top_n_tuples])

            prob_integer_pos = (
                math.comb(60, 4) - math.comb(60 - top_n, 4)
            ) / math.comb(60, 4)
            joint_occurrences = top_n_sum
            joint_prob = joint_occurrences / total_obs
            joint_self_inf = -np.log2(joint_prob)
            npmi = np.log2(joint_prob / (msg_prob * prob_integer_pos)) / joint_self_inf
            non_compositional_npmi_dict[msg][pos] = {
                "npmi": npmi,
                "ints": top_n_keys,
            }

    return non_compositional_npmi_dict


# How to do this?
# Calculate the normalised pointwise mutual information
# Between all possible n-grams of messages
# Over all possible integers
# This is the first step to decoding the messages
def compute_compositional_ngrams_integers_npmi(  # noqa: C901
    match, n_grams, top_n: int = 5
) -> (dict, dict):
    """
    Compute the compositional NPMI measure for the possible integer n-grams.

    Parameters
    ----------
    match: dict
        Dictionary of matches. For format please see analysis.ipynb.
    n_grams: dict
        Dictionary of all possible n-grams and their lengths. For format please see analysis.ipynb.
    top_n: int
        Number of integer candidates to consider when computing compositional NPMI.

    Returns
    -------
    Compositional NPMI for n-grams as integer reference candidates and pruned n-grams for later computation
    """
    match_tpg_dict = match["tpg_stats"]
    total_messages = len(match["cut_inputs"])

    n_grams_pruned = copy.deepcopy(n_grams)

    for n_gram in n_grams.keys():
        n_gram_in = False
        n_gram_len = n_grams_pruned[n_gram]
        n_grams_pruned[n_gram] = {
            x: {
                "total": 0,
                "integers": {str(y): 0 for y in range(60)},
            }
            for x in [0, 1, 2]
        }
        n_grams_pruned[n_gram]["length"] = n_gram_len
        n_gram_dec = np.array([x for x in n_gram.split(" ")])
        # Get all messages with this ngram in given position
        for msg in match_tpg_dict.keys():
            msg_dec = [
                x for x in msg.replace("[", "").replace("]", "").strip().split(" ") if x
            ]
            if n_gram_len == 1:
                for idx, msg_d in enumerate(msg_dec):
                    if msg_d == n_gram:
                        n_gram_in = True
                        n_grams_pruned[n_gram][idx]["total"] += match_tpg_dict[msg][
                            "total"
                        ]
                        for pos in match_tpg_dict[msg]["obs_neighbours"]:
                            for neighbour in match_tpg_dict[msg]["obs_neighbours"][pos]:
                                n_grams_pruned[n_gram][idx]["integers"][
                                    neighbour
                                ] += match_tpg_dict[msg]["obs_neighbours"][pos][
                                    neighbour
                                ]
            elif n_gram_len == 2:
                first = False
                second = False
                both = False
                if n_gram_dec[0] == msg_dec[0] and n_gram_dec[1] == msg_dec[1]:
                    first = True
                if n_gram_dec[0] == msg_dec[1] and n_gram_dec[1] == msg_dec[2]:
                    second = True
                if first and second:
                    both = True

                if first or second:
                    if both:
                        n_gram_in = True
                        for indx in [0, 1]:
                            n_grams_pruned[n_gram][indx]["total"] += match_tpg_dict[
                                msg
                            ]["total"]
                            for pos in match_tpg_dict[msg]["obs_neighbours"]:
                                for neighbour in match_tpg_dict[msg]["obs_neighbours"][
                                    pos
                                ]:
                                    n_grams_pruned[n_gram][indx]["integers"][
                                        neighbour
                                    ] += match_tpg_dict[msg]["obs_neighbours"][pos][
                                        neighbour
                                    ]
                    else:
                        if second:  # Last 2 chars of message
                            n_gram_in = True
                            indx = 1
                            n_grams_pruned[n_gram][indx]["total"] += match_tpg_dict[
                                msg
                            ]["total"]
                            for pos in match_tpg_dict[msg]["obs_neighbours"]:
                                for neighbour in match_tpg_dict[msg]["obs_neighbours"][
                                    pos
                                ]:
                                    n_grams_pruned[n_gram][indx]["integers"][
                                        neighbour
                                    ] += match_tpg_dict[msg]["obs_neighbours"][pos][
                                        neighbour
                                    ]
                        elif first:  # First 2 chars of message
                            n_gram_in = True
                            indx = 0
                            n_grams_pruned[n_gram][indx]["total"] += match_tpg_dict[
                                msg
                            ]["total"]
                            for pos in match_tpg_dict[msg]["obs_neighbours"]:
                                for neighbour in match_tpg_dict[msg]["obs_neighbours"][
                                    pos
                                ]:
                                    n_grams_pruned[n_gram][indx]["integers"][
                                        neighbour
                                    ] += match_tpg_dict[msg]["obs_neighbours"][pos][
                                        neighbour
                                    ]
                        else:
                            pass

            else:
                if (
                    n_gram_dec[0] == msg_dec[0]
                    and n_gram_dec[1] == msg_dec[1]
                    and n_gram_dec[2] == msg_dec[2]
                ):
                    n_gram_in = True
                    indx = 0
                    n_grams_pruned[n_gram][indx]["total"] += match_tpg_dict[msg][
                        "total"
                    ]
                    for pos in match_tpg_dict[msg]["obs_neighbours"]:
                        for neighbour in match_tpg_dict[msg]["obs_neighbours"][pos]:
                            n_grams_pruned[n_gram][indx]["integers"][
                                neighbour
                            ] += match_tpg_dict[msg]["obs_neighbours"][pos][neighbour]

        if not n_gram_in:
            del n_grams_pruned[n_gram]

    # Calculate NPMI for each integer
    ngram_npmi_integers_dict = {}
    for n_gram in n_grams_pruned:
        # Also calculate position invariant by combining the counts
        ngram_npmi_integers_dict[n_gram] = {f"npmi_pos_{x}": {} for x in [0, 1, 2]}
        ngram_npmi_integers_dict[n_gram]["inv_npmi"] = {}
        combined_count_ngram = 0
        combined_count_integers = {str(y): 0 for y in range(60)}

        # Probability of choosing any of the top_n integers, without replacement
        # This is dependent on the sequence window size!
        prob_integer_pos = (math.comb(60, 4) - math.comb(60 - top_n, 4)) / math.comb(
            60, 4
        )

        for pos in [0, 1, 2]:
            # Get the highest occurrences integers
            integers_dict = n_grams_pruned[n_gram][pos]["integers"]
            top_n_tuples = sorted(
                integers_dict.items(), key=lambda pair: pair[1], reverse=True
            )[:top_n]
            top_n_keys = [x[0] for x in top_n_tuples]
            top_n_sum = sum([x[1] for x in top_n_tuples])

            # Add to the position invariant calculation
            combined_count_ngram += n_grams_pruned[n_gram][pos]["total"]
            for integer_count_tuple in top_n_tuples:
                combined_count_integers[integer_count_tuple[0]] += integer_count_tuple[
                    1
                ]

            # ngram unused in this position
            if top_n_sum == 0:
                continue

            n_gram_total_count = n_grams_pruned[n_gram][pos]["total"]
            prob_n_gram_pos = n_gram_total_count / total_messages
            prob_n_gram_integer_pos = top_n_sum / total_messages

            pmi = np.log2(
                prob_n_gram_integer_pos / (prob_n_gram_pos * prob_integer_pos)
            )
            h = -np.log2(prob_n_gram_integer_pos)
            npmi = pmi / h
            ngram_npmi_integers_dict[n_gram][f"npmi_pos_{pos}"]["value"] = npmi
            ngram_npmi_integers_dict[n_gram][f"npmi_pos_{pos}"]["integers"] = top_n_keys
            if npmi > 1:
                ngram_npmi_integers_dict[n_gram][f"npmi_pos_{pos}"]["problem"] = {
                    "pmi": pmi,
                    "prob_ngram": prob_n_gram_pos,
                    "prob_integer": prob_integer_pos,
                    "prob_n_gram_integer": prob_n_gram_integer_pos,
                    "h": h,
                }

        top_n_tuples = sorted(
            combined_count_integers.items(), key=lambda pair: pair[1], reverse=True
        )[:top_n]
        top_n_keys = [x[0] for x in top_n_tuples]
        top_n_sum = sum([x[1] for x in top_n_tuples])
        prob_n_gram = combined_count_ngram / (
            total_messages * (4 - n_grams_pruned[n_gram]["length"])
        )
        prob_integer = prob_integer_pos
        prob_n_gram_integer = top_n_sum / (
            total_messages * (4 - n_grams_pruned[n_gram]["length"])
        )

        pmi = np.log2(prob_n_gram_integer / (prob_n_gram * prob_integer))
        h = -np.log2(prob_n_gram_integer)
        npmi = pmi / h
        if npmi > 1:
            ngram_npmi_integers_dict[n_gram]["inv_npmi"]["problem"] = {
                "pmi": pmi,
                "prob_ngram": prob_n_gram,
                "prob_integer": prob_integer,
                "prob_n_gram_integer": prob_n_gram_integer,
                "h": h,
            }
        ngram_npmi_integers_dict[n_gram]["inv_npmi"]["value"] = npmi
        ngram_npmi_integers_dict[n_gram]["inv_npmi"]["integers"] = top_n_keys
        ngram_npmi_integers_dict[n_gram]["inv_npmi"]["data"] = n_grams_pruned[n_gram]

    return ngram_npmi_integers_dict, n_grams_pruned


# Calculate the normalised pointwise mutual information
# Between all identified meaningful ngrams
# And the positions where their targets may be
# This is the second and last step to decoding the messages
def compute_compositional_ngrams_positionals_npmi(  # noqa: C901
    match: dict, n_grams: dict, confidence: float, top_n: int, scale: int = 1
) -> dict:
    """
    Compute the compositional NPMI, for the possible positional ngrams.

    Requires the match to already contain the integer ngrams.

    Parameters
    ----------
    match: dict
        Dictionary of matches. For format please see analysis.ipynb.
    n_grams: dict
        Dictionary of all possible n-grams and their lengths. For format please see analysis.ipynb.
    confidence: float
        Cut-off point for when to consider an integer candidate as a reference. Must be between 0-1.
    top_n: int
        Number of integer candidates to consider when computing compositional NPMI.
    scale: int
        What amount to scale the NPMI by. Default 1 (no scaling).

    Returns
    -------
    Dictionary of candidate positional n-grams.
    """
    ngram_npmi_integers_dict = match[f"ngram_npmi_integers_{top_n}"]

    identified_ngrams = defaultdict(dict)
    ngram_npmi_positionals_dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    match_tpg_dict = match["tpg_stats"]
    total_messages = total_obs = len(match["cut_inputs"])

    # Purge ngrams which are not quite certain
    for ngram in ngram_npmi_integers_dict:
        # Skip n-grams size 3, they cannot contain more information
        # As they would be non-compositional so accounted in the previous algo
        if n_grams[ngram] == 3:
            continue
        for pos in ngram_npmi_integers_dict[ngram]:
            if len(ngram_npmi_integers_dict[ngram][pos]) == 0:
                continue
            if ngram_npmi_integers_dict[ngram][pos]["value"] > confidence:
                identified_ngrams[ngram][pos] = {
                    "integers": ngram_npmi_integers_dict[ngram][pos]["integers"],
                }

    for ngram in identified_ngrams:
        n_gram_dec = np.array([x for x in ngram.split(" ")])
        n_gram_len = n_grams[ngram]
        n_gram_in = False

        # Skip n-grams size 3, they cannot contain more information
        # As they would be non-compositional so accounted in the previous algo
        if n_gram_len == 3:
            continue

        for msg in match_tpg_dict:
            msg_dec = [
                x for x in msg.replace("[", "").replace("]", "").strip().split(" ") if x
            ]
            for pos_int_ngram in identified_ngrams[ngram]:
                n_gram_mid = False
                msg_c = copy.deepcopy(msg_dec)
                if pos_int_ngram == "inv_npmi":
                    # Proceed with position invariant analysis
                    if n_gram_len == 1:
                        n_gram_first = n_gram_dec[0] == msg_dec[0]
                        n_gram_mid = n_gram_dec[0] == msg_dec[1]
                        n_gram_last = n_gram_dec[0] == msg_dec[2]
                        n_gram_in = n_gram_first or n_gram_mid or n_gram_last

                        if n_gram_first:
                            msg_c.pop(0)
                        elif n_gram_mid:
                            msg_c.pop(1)
                        elif n_gram_last:
                            msg_c.pop(2)

                    if n_gram_len == 2:
                        first = (
                            n_gram_dec[0] == msg_dec[0] and n_gram_dec[1] == msg_dec[1]
                        )
                        second = (
                            n_gram_dec[0] == msg_dec[1] and n_gram_dec[1] == msg_dec[2]
                        )
                        n_gram_in = first or second

                        if first:
                            msg_c = [msg_dec[2]]
                        elif second:
                            msg_c = [msg_dec[0]]

                    if n_gram_in:
                        for integer in identified_ngrams[ngram][pos_int_ngram][
                            "integers"
                        ]:
                            for neighbour_pos in match_tpg_dict[msg]["obs_neighbours"]:
                                if (
                                    integer
                                    in match_tpg_dict[msg]["obs_neighbours"][
                                        neighbour_pos
                                    ].keys()
                                ):
                                    # Iterate over all leftover message characters,
                                    # in case some characters are just fillers
                                    for msg_ngram in msg_c:
                                        msg_ngram = str(msg_ngram)
                                        ngram_npmi_positionals_dict[msg_ngram][
                                            "inv_npmi"
                                        ][neighbour_pos] += match_tpg_dict[msg][
                                            "obs_neighbours"
                                        ][
                                            neighbour_pos
                                        ][
                                            integer
                                        ]
                                        # Also account for lack of precision in agents' language
                                        if "r" in neighbour_pos:
                                            ngram_npmi_positionals_dict[msg_ngram][
                                                "inv_npmi"
                                            ]["r"] += match_tpg_dict[msg][
                                                "obs_neighbours"
                                            ][
                                                neighbour_pos
                                            ][
                                                integer
                                            ]
                                        else:
                                            ngram_npmi_positionals_dict[msg_ngram][
                                                "inv_npmi"
                                            ]["l"] += match_tpg_dict[msg][
                                                "obs_neighbours"
                                            ][
                                                neighbour_pos
                                            ][
                                                integer
                                            ]
                                    if len(msg_c) > 1:
                                        if n_gram_mid:
                                            pass
                                        else:
                                            msg_c_str = str(msg_c)
                                            ngram_npmi_positionals_dict[msg_c_str][
                                                "inv_npmi"
                                            ][neighbour_pos] += match_tpg_dict[msg][
                                                "obs_neighbours"
                                            ][
                                                neighbour_pos
                                            ][
                                                integer
                                            ]
                                            # Also account for lack of precision in agents' language
                                            if "r" in neighbour_pos:
                                                ngram_npmi_positionals_dict[msg_c_str][
                                                    "inv_npmi"
                                                ]["r"] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                                            else:
                                                ngram_npmi_positionals_dict[msg_c_str][
                                                    "inv_npmi"
                                                ]["l"] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                else:
                    match pos_int_ngram:
                        case "npmi_pos_0":
                            if n_gram_len == 1:
                                n_gram_in = n_gram_dec[0] == msg_dec[0]
                                msg_c.pop(0)
                            if n_gram_len == 2:
                                n_gram_in = (
                                    n_gram_dec[0] == msg_dec[0]
                                    and n_gram_dec[1] == msg_dec[1]
                                )
                                msg_c = [msg_dec[2]]

                            if n_gram_in:
                                for integer in identified_ngrams[ngram][pos_int_ngram][
                                    "integers"
                                ]:
                                    for neighbour_pos in match_tpg_dict[msg][
                                        "obs_neighbours"
                                    ]:
                                        if (
                                            integer
                                            in match_tpg_dict[msg]["obs_neighbours"][
                                                neighbour_pos
                                            ].keys()
                                        ):
                                            for idx, msg_ngram in zip([2, 1], msg_c):
                                                msg_ngram = str(msg_ngram)
                                                ngram_npmi_positionals_dict[msg_ngram][
                                                    f"npmi_pos_{idx}"
                                                ][neighbour_pos] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                                                # Also account for lack of precision in agents' language
                                                if "r" in neighbour_pos:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "r"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                                else:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "l"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                            if len(msg_c) > 1:
                                                msg_c_str = str(msg_c)
                                                ngram_npmi_positionals_dict[msg_c_str][
                                                    f"npmi_pos_{1}"
                                                ][neighbour_pos] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                                                # Also account for lack of precision in agents' language
                                                if "r" in neighbour_pos:
                                                    ngram_npmi_positionals_dict[
                                                        msg_c_str
                                                    ][f"npmi_pos_{1}"][
                                                        "r"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                                else:
                                                    ngram_npmi_positionals_dict[
                                                        msg_c_str
                                                    ][f"npmi_pos_{1}"][
                                                        "l"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]

                        case "npmi_pos_1":
                            if n_gram_len == 1:
                                n_gram_in = n_gram_dec[0] == msg_dec[1]
                                msg_c.pop(1)
                            if n_gram_len == 2:
                                n_gram_in = (
                                    n_gram_dec[0] == msg_dec[1]
                                    and n_gram_dec[1] == msg_dec[2]
                                )
                                msg_c = [msg_dec[0]]

                            if n_gram_in:
                                for integer in identified_ngrams[ngram][pos_int_ngram][
                                    "integers"
                                ]:
                                    for neighbour_pos in match_tpg_dict[msg][
                                        "obs_neighbours"
                                    ]:
                                        if (
                                            integer
                                            in match_tpg_dict[msg]["obs_neighbours"][
                                                neighbour_pos
                                            ].keys()
                                        ):
                                            for idx, msg_ngram in zip([0, 2], msg_c):
                                                msg_ngram = str(msg_ngram)
                                                ngram_npmi_positionals_dict[msg_ngram][
                                                    f"npmi_pos_{idx}"
                                                ][neighbour_pos] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                                                # Also account for lack of precision in agents' language
                                                if "r" in neighbour_pos:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "r"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                                else:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "l"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]

                        case "npmi_pos_2":
                            if n_gram_len == 1:
                                n_gram_in = n_gram_dec[0] == msg_dec[2]
                                msg_c.pop(2)
                            # Cannot have n_gram of len 2 in pos 2
                            if n_gram_len == 2:
                                n_gram_in = False
                                raise ValueError("Incorrect ngram dict!")

                            if n_gram_in:
                                for integer in identified_ngrams[ngram][pos_int_ngram][
                                    "integers"
                                ]:
                                    for neighbour_pos in match_tpg_dict[msg][
                                        "obs_neighbours"
                                    ]:
                                        if (
                                            integer
                                            in match_tpg_dict[msg]["obs_neighbours"][
                                                neighbour_pos
                                            ].keys()
                                        ):
                                            for idx, msg_ngram in zip([0, 1], msg_c):
                                                msg_ngram = str(msg_ngram)
                                                ngram_npmi_positionals_dict[msg_ngram][
                                                    f"npmi_pos_{idx}"
                                                ][neighbour_pos] += match_tpg_dict[msg][
                                                    "obs_neighbours"
                                                ][
                                                    neighbour_pos
                                                ][
                                                    integer
                                                ]
                                                # Also account for lack of precision in agents' language
                                                if "r" in neighbour_pos:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "r"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                                else:
                                                    ngram_npmi_positionals_dict[
                                                        msg_ngram
                                                    ][f"npmi_pos_{idx}"][
                                                        "l"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                            if len(msg_c) > 1:
                                                msg_c_str = str(msg_c)
                                                ngram_npmi_positionals_dict[msg_c_str][
                                                    f"npmi_pos_{0}"
                                                ][neighbour_pos] += 1
                                                # Also account for lack of precision in agents' language
                                                if "r" in neighbour_pos:
                                                    ngram_npmi_positionals_dict[
                                                        msg_c_str
                                                    ][f"npmi_pos_{0}"][
                                                        "r"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]
                                                else:
                                                    ngram_npmi_positionals_dict[
                                                        msg_c_str
                                                    ][f"npmi_pos_{0}"][
                                                        "l"
                                                    ] += match_tpg_dict[
                                                        msg
                                                    ][
                                                        "obs_neighbours"
                                                    ][
                                                        neighbour_pos
                                                    ][
                                                        integer
                                                    ]

    # Compute NPMI for all possible positional ngrams
    ngram_npmi_positionals_clean_dict = {}

    n_grams_pruned = match["ngrams_pruned"]

    for n_gram in ngram_npmi_positionals_dict:
        # Clean up the ngram representation
        n_gram_str = (
            n_gram.replace("[", "").replace("]", "").replace(",", "").replace("'", "")
        )
        ngram_npmi_positionals_clean_dict[n_gram_str] = {
            f"npmi_pos_{x}": {} for x in [0, 1, 2]
        }
        ngram_npmi_positionals_clean_dict[n_gram_str]["inv_npmi"] = {}

        n_gram_total_count = 0

        # Probabilities of a given position existing in the observation
        prob_pos = {
            "r": (total_obs - match["obs_counts"]["end"]) / total_obs,
            "l": (total_obs - match["obs_counts"]["begin"]) / total_obs,
            "l1": (total_obs - match["obs_counts"]["begin"]) / total_obs,
            "l2": (
                total_obs
                - match["obs_counts"]["begin"]
                - match["obs_counts"]["begin+1"]
            )
            / total_obs,
            "l3": (match["obs_counts"]["end"] + match["obs_counts"]["end-1"])
            / total_obs,
            "l4": match["obs_counts"]["end"] / total_obs,
            "r1": (total_obs - match["obs_counts"]["end"]) / total_obs,
            "r2": (
                total_obs - match["obs_counts"]["end"] - match["obs_counts"]["end-1"]
            )
            / total_obs,
            "r3": (match["obs_counts"]["begin"] + match["obs_counts"]["begin+1"])
            / total_obs,
            "r4": match["obs_counts"]["begin"] / total_obs,
        }

        # Loop over all ngram positions
        # Position variant calculations
        for ngram_pos in [0, 1, 2]:
            n_gram_pos_count = n_grams_pruned[n_gram_str][ngram_pos]["total"]
            n_gram_total_count += n_gram_pos_count
            prob_n_gram_in_pos = n_gram_pos_count / total_messages
            for referent_pos in ngram_npmi_positionals_dict[n_gram][
                f"npmi_pos_{ngram_pos}"
            ]:
                # Skip if n_gram never in position
                if n_gram_pos_count == 0:
                    ngram_npmi_positionals_clean_dict[n_gram_str][
                        f"npmi_pos_{ngram_pos}"
                    ][
                        referent_pos
                    ] = -2  # Invalid
                    continue

                prob_given_pos = prob_pos[referent_pos]
                count_joint_ngram_given_pos = ngram_npmi_positionals_dict[n_gram][
                    f"npmi_pos_{ngram_pos}"
                ][referent_pos]
                prob_joint_ngram_given_pos = count_joint_ngram_given_pos / total_obs
                joint_self_inf = -np.log2(prob_joint_ngram_given_pos)
                npmi = (
                    np.log2(
                        prob_joint_ngram_given_pos
                        / (prob_n_gram_in_pos * prob_given_pos)
                    )
                    / joint_self_inf
                )
                ngram_npmi_positionals_clean_dict[n_gram_str][f"npmi_pos_{ngram_pos}"][
                    referent_pos
                ] = npmi

        # Position invariant calculations
        prob_n_gram_total = n_gram_total_count / total_messages
        for referent_pos_inv in ngram_npmi_positionals_dict[n_gram]["inv_npmi"]:
            prob_given_pos = prob_pos[referent_pos_inv]
            count_joint_ngram_given_pos = ngram_npmi_positionals_dict[n_gram][
                "inv_npmi"
            ][referent_pos_inv]
            prob_joint_ngram_given_pos = count_joint_ngram_given_pos / total_obs
            joint_self_inf = -np.log2(prob_joint_ngram_given_pos)
            npmi = (
                np.log2(
                    prob_joint_ngram_given_pos / (prob_n_gram_total * prob_given_pos)
                )
                / joint_self_inf
            )
            ngram_npmi_positionals_clean_dict[n_gram_str]["inv_npmi"][
                referent_pos_inv
            ] = npmi

    # Clean up and optionally scale the dict
    # The .keys() thing is because with deletion python will complain about iterable changing size
    ngram_npmi_positionals_clean_dict_keys = list(
        ngram_npmi_positionals_clean_dict.keys()
    )
    for ngram in ngram_npmi_positionals_clean_dict_keys:
        ngram_npmi_positionals_clean_dict_pos_keys = list(
            ngram_npmi_positionals_clean_dict[ngram].keys()
        )
        for ngram_pos in ngram_npmi_positionals_clean_dict_pos_keys:
            ngram_npmi_positionals_clean_dict_ref_keys = list(
                ngram_npmi_positionals_clean_dict[ngram][ngram_pos].keys()
            )
            for referent_pos in ngram_npmi_positionals_clean_dict_ref_keys:
                if scale:
                    ngram_npmi_positionals_clean_dict[ngram][ngram_pos][
                        referent_pos
                    ] *= 10
                if (
                    ngram_npmi_positionals_clean_dict[ngram][ngram_pos][referent_pos]
                    < 0
                ):
                    del ngram_npmi_positionals_clean_dict[ngram][ngram_pos][
                        referent_pos
                    ]

            if len(ngram_npmi_positionals_clean_dict[ngram][ngram_pos]) == 0:
                del ngram_npmi_positionals_clean_dict[ngram][ngram_pos]

        if len(ngram_npmi_positionals_clean_dict[ngram]) == 0:
            del ngram_npmi_positionals_clean_dict[ngram]

    return ngram_npmi_positionals_clean_dict


# if __name__ == "__main__":
#     import itertools
#     import pickle
#
#     n_grams = defaultdict(dict)
#     for x in [1, 2, 3]:
#         for n_gram in list(itertools.product([x for x in range(26)], repeat=x)):
#             n_grams[n_gram]["length"] = x
#     n_grams = {
#         str(n_gram)
#         .replace("(", "")
#         .replace(")", "")
#         .replace(",", ""): n_grams[n_gram]["length"]
#         for n_gram in n_grams.keys()
#     }
#     with open("../../analysis/matches_debug.pickle", "rb") as handle:
#         matches = pickle.load(handle)
#     a = compute_compositional_ngrams_positionals_npmi(
#         match=matches["match_1"], n_grams=n_grams, confidence=0.5, top_n=1, scale=10
#     )
