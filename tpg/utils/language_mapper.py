"""Module allowing for mapping observations to messages identified through the analysis."""
from typing import Tuple

import numpy as np
import torch
from numpy.random import default_rng


class LanguageMapper:
    """The class that implements the mapper from observations to a language that agents can understand."""

    def __init__(
        self,
        message_length: int = 3,
        vocab_size: int = 26,
        positional_message_dict: dict = None,
        non_compo_message_dict: dict = None,
        compo_message_dict_int: dict = None,
        compo_message_dict_pos: dict = None,
    ):
        """
        Initialise the language mapper, and create the language mapping.

        Any observations that do not have a corresponding messages will be assigned a random message.

        Parameters
        ----------
        message_length: int
            Maximum length of a message.
        vocab_size: int
            Size of the agents' vocabulary.
        positional_message_dict: dict
            Dictionary containing a mapping of each positional function ("begin","begin+1","end-1","end") to a message.
            The format of this dict is {function: message}.
        non_compo_message_dict: dict
            Dictionary containing a mapping of non-compositional message to their meanings.
            The format of this dict is {pos: {int: message}}, where pos refers to [l1,l2,l3,...], and
            int is the integer in that position.
        compo_message_dict_int: dict
            Dictionary containing a mapping of atomic parts of compositional messages to their integer meanings.
        compo_message_dict_pos: dict
            Dictionary containing a mapping of atomic parts of compositional messages to their positional meanings.
        """
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.positional_message_dict = positional_message_dict
        self.non_compo_message_dict = non_compo_message_dict
        self.compo_message_dict_int = compo_message_dict_int
        self.compo_message_dict_pos = compo_message_dict_pos

        self.compo = True if compo_message_dict_int else False
        self.compo_pos = True if compo_message_dict_pos else False
        self.non_compo = True if non_compo_message_dict else False
        self.positional = True if positional_message_dict else False

        self.compo_pos_failures = 0

        self.rng = default_rng()

        if positional_message_dict is not None:
            for key in positional_message_dict:
                if key not in [
                    "begin",
                    "begin+1",
                    "end-1",
                    "end",
                ]:
                    raise ValueError("Unknown positional message type.")
        else:
            self.positional_message_dict = {}

        if non_compo_message_dict is not None:
            for key in non_compo_message_dict:
                if key not in [
                    "l1",
                    "l2",
                    "l3",
                    "l4",
                    "r1",
                    "r2",
                    "r3",
                    "r4",
                ]:
                    raise ValueError("Unknown other message type.")
        else:
            self.non_compo_message_dict = {}

        self.language = self.create_language()
        self.reverse_lang = {}

        for k, v in self.language.items():
            if k in [
                "begin",
                "begin+1",
                "end-1",
                "end",
            ]:
                if len(v) > 1:
                    for idx, msg in enumerate(v):
                        self.reverse_lang[str(msg)] = f"{k}_{idx}"
                else:
                    self.reverse_lang[str(v[0])] = f"{k}_0"

        for pos in non_compo_message_dict:
            for integer in non_compo_message_dict[pos]:
                v = non_compo_message_dict[pos][integer]
                if len(v) > 1:
                    for idx, msg in enumerate(v):
                        self.reverse_lang[str(msg)] = f"{pos}-{integer}_{idx}"
                else:
                    self.reverse_lang[str(v[0])] = f"{pos}-{integer}_0"

    def create_language(self):
        """Create the language based on the provided dictionaries."""
        language = dict()
        for key in [
            "begin",
            "begin+1",
            "end-1",
            "end",
        ]:
            if (
                key in self.positional_message_dict
                and len(self.positional_message_dict[key]) > 0
            ):
                language[key] = np.array(self.positional_message_dict[key])

        # Copy the nc dict almost verbatim to language dict
        for pos in self.non_compo_message_dict:
            language[pos] = {
                integer: self.non_compo_message_dict[pos][integer]
                for integer in self.non_compo_message_dict[pos]
            }

        # Copy the integer compositional dict
        for npmi_pos in self.compo_message_dict_int:
            if "inv" not in npmi_pos:
                pos = npmi_pos.split("_")[2]
            else:
                pos = "inv"

            language[pos] = {
                integer: [
                    msg for msg in self.compo_message_dict_int[npmi_pos][integer]
                ]  # Ignore length 3 n-grams, as they are accounted for in the Noncompo analysis
                for integer in self.compo_message_dict_int[npmi_pos]
            }

            # Prune empty lists
            for integer in self.compo_message_dict_int[npmi_pos]:
                if not language[pos][integer]:
                    del language[pos][integer]

        return language

    def map_obs_message(
        self, obs: np.ndarray | torch.Tensor
    ) -> Tuple[Tuple[np.ndarray, list], np.ndarray]:
        """
        Map an observation to a message.

        Will either use a message from the dictionary, or generate a random message if
        observation cannot be mapped to a known message, or n-gram combination.

        Parameters
        ----------
        obs: np.ndarray
            A 1-D or 2-D array of observation(s) to be mapped to messages.
        """
        meanings = []
        messages = []
        if obs.ndim == 1:
            if self.compo:
                messages.append(self.map_obs_helper(obs)[0])
                meanings.append(self.map_obs_helper(obs)[1])
            else:
                messages.append(self.map_obs_helper(obs))
        elif obs.ndim == 2:
            for observation in obs:
                if self.compo:
                    messages.append(self.map_obs_helper(observation)[0])
                    meanings.append(self.map_obs_helper(observation)[1])
                else:
                    messages.append(self.map_obs_helper(observation))
        else:
            raise ValueError("Cannot accept 3D observations!")

        messages = np.array(messages)

        # Agents expect a one-hot message
        one_hot_messages = []
        for message in messages:
            message_oh = np.eye(self.vocab_size)[message.reshape(-1)]
            one_hot_messages.append(message_oh)

        return (messages, meanings), np.array(one_hot_messages)

    def map_obs_helper(self, observation: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Map an observation to a message.

        Will either use a message from the dictionary, or generate a random message if
        observation cannot be mapped to a known message, or n-gram combination.

        Parameters
        ----------
        observation: np.ndarray
            A 1-D array for a single observation to be mapped to messages.
        """
        idx = np.where(observation == -1)[0]
        integers = list(observation)
        match idx:
            case 0:
                # Check r1-r4
                message = (
                    self.non_compo_helper(
                        positions={"r1": 1, "r2": 2, "r3": 3, "r4": 4},
                        integers=integers,
                    )
                    if self.non_compo
                    else None
                )
                message_c = (
                    self.compo_helper(observation=observation) if self.compo else None
                )
                # Always try the positional messages first
                if "begin" in self.language:
                    if len(self.language["begin"]) > 1:
                        return self.rng.choice(self.language["begin"])
                    else:
                        return self.language["begin"][0]
                elif message is not None:
                    return message
                elif message_c is not None:
                    return message_c
                else:
                    return self.rng.choice(
                        self.vocab_size, size=self.message_length, replace=True
                    )
            case 1:
                # Check l1,r1,r2,r3
                message = (
                    self.non_compo_helper(
                        positions={"l1": 0, "r1": 2, "r2": 3, "r3": 4},
                        integers=integers,
                    )
                    if self.non_compo
                    else None
                )
                message_c = (
                    self.compo_helper(observation=observation) if self.compo else None
                )
                # Always try the positional messages first
                if "begin+1" in self.language:
                    if len(self.language["begin+1"]) > 1:
                        return self.rng.choice(self.language["begin+1"])
                    else:
                        return self.language["begin+1"][0]
                elif message is not None:
                    return message
                elif message_c is not None:
                    return message_c
                else:
                    return self.rng.choice(
                        self.vocab_size, size=self.message_length, replace=True
                    )
            case 2:
                # Check l1,l2,r1,r2
                # First check non-compositional messages.
                # If agents have agreed on a non-compositional message
                # for a given observation, then they probably expect that message.
                message = (
                    self.non_compo_helper(
                        positions={"l1": 1, "l2": 0, "r1": 3, "r2": 4},
                        integers=integers,
                    )
                    if self.non_compo
                    else None
                )
                message_c = (
                    self.compo_helper(observation=observation) if self.compo else None
                )
                if message is not None:
                    return message
                elif message_c is not None:
                    return message_c
                else:
                    return self.rng.choice(
                        self.vocab_size, size=self.message_length, replace=True
                    )
            case 3:
                # Check r1,l1,l2,l3
                message = (
                    self.non_compo_helper(
                        positions={"l1": 2, "l2": 1, "l3": 0, "r1": 4},
                        integers=integers,
                    )
                    if self.non_compo
                    else None
                )
                message_c = (
                    self.compo_helper(observation=observation) if self.compo else None
                )
                # Always try the positional messages first
                if "end-1" in self.language:
                    if len(self.language["end-1"]) > 1:
                        return self.rng.choice(self.language["end-1"])
                    else:
                        return self.language["end-1"][0]
                elif message is not None:
                    return message
                elif message_c is not None:
                    return message_c
                else:
                    return self.rng.choice(
                        self.vocab_size, size=self.message_length, replace=True
                    )
            case 4:
                # Check l1,l2,l3,l4
                message = (
                    self.non_compo_helper(
                        positions={"l1": 3, "l2": 2, "l3": 1, "l4": 0},
                        integers=integers,
                    )
                    if self.non_compo
                    else None
                )
                message_c = (
                    self.compo_helper(observation=observation) if self.compo else None
                )
                # Always try the positional messages first
                if "end" in self.language:
                    if len(self.language["end"]) > 1:
                        return self.rng.choice(self.language["end"])
                    else:
                        return self.language["end"][0]
                elif message is not None:
                    return message
                elif message_c is not None:
                    return message_c
                else:
                    return self.rng.choice(
                        self.vocab_size, size=self.message_length, replace=True
                    )

    def non_compo_helper(self, positions: dict, integers: list):
        """
        Help mapping non-compositional messages to an observation.

        Parameters
        ----------
        positions: list
            Positions of the integer to check.
        integers: list
            Integers to check.
        """
        # check if any position is in agents' language
        positions_keys = list(positions.keys())
        self.rng.shuffle(positions_keys)
        for pos in positions_keys:
            if pos in self.language:
                integer = int(integers[positions[pos]])
                if integer in self.language[pos]:
                    # if more than one pick randomly
                    if len(self.language[pos][integer]) > 1:
                        # TODO return meanings here as well, as the messages can be reverse mapped to multiple things
                        # So the reverse lang label is then incorrect
                        return self.rng.choice(self.language[pos][integer])
                    else:
                        return self.language[pos][integer][0]

        return None

    def compo_helper(  # noqa: C901
        self, observation: np.ndarray | torch.Tensor
    ) -> (np.ndarray, str):
        """
        Help creating a compositional message from an observation.

        Parameters
        ----------
        observation: np.ndarray | torch.Tensor
            Observation to create a message for
        """
        msg = [-1, -1, -1]
        filled_ints = []
        filled_pos = []
        reverse_lang_dict = {
            "ints": [],
            "integer_pos": [],
            "ngram_pos": [],
            "msg_pos": [],
            "ngrams": [],
        }
        meaning = None
        found = False

        if self.compo_pos:
            found, msg_components = self._compo_pos_helper(observation=observation)
            # assemble message from components
            if found:
                referent_position_pos = msg_components[0]
                ngram_ref_pos = msg_components[1]
                integer_pos_msg = msg_components[2]
                ngram_int = msg_components[3]
                ref_integer = msg_components[4]
                ref_pos = msg_components[5]

                integer_in = False
                ref_pos_in = False

                for possible_msg_pos in [0, 1, 2]:
                    # This is not right!!!
                    if f"{possible_msg_pos}" in referent_position_pos:
                        ref_pos_in = True
                        if len(ngram_ref_pos) == 2:
                            filled_pos.append(possible_msg_pos)
                            filled_pos.append(possible_msg_pos + 1)
                            msg[possible_msg_pos] = ngram_ref_pos[0]
                            msg[possible_msg_pos + 1] = ngram_ref_pos[1]
                        else:
                            filled_pos.append(possible_msg_pos)
                            msg[possible_msg_pos] = ngram_ref_pos[0]
                    if f"{possible_msg_pos}" in integer_pos_msg:
                        integer_in = True
                        if len(ngram_int) == 2:
                            filled_pos.append(possible_msg_pos)
                            filled_pos.append(possible_msg_pos + 1)
                            msg[possible_msg_pos] = ngram_int[0]
                            msg[possible_msg_pos + 1] = ngram_int[1]
                        else:
                            filled_pos.append(possible_msg_pos)
                            msg[possible_msg_pos] = ngram_int[0]

                # At least one of the ngrams must be position invariant
                missing_poses = [x for x in [0, 1, 2] if x not in filled_pos]
                self.rng.shuffle(missing_poses)
                for missing_position in missing_poses:
                    if not integer_in:
                        if len(ngram_int) == 2:
                            if missing_position == 2:
                                continue
                            if missing_position == 1:
                                if 2 not in missing_poses:
                                    continue
                            filled_pos.append(missing_position)
                            filled_pos.append(missing_position + 1)
                            msg[missing_position] = ngram_int[0]
                            msg[missing_position + 1] = ngram_int[1]
                            integer_in = True
                        else:
                            filled_pos.append(missing_position)
                            msg[missing_poses[0]] = ngram_int[0]
                            integer_in = True
                    if not ref_pos_in:
                        if len(ngram_ref_pos) == 2:
                            if missing_position == 2:
                                continue
                            if missing_position == 1:
                                if 2 not in missing_poses:
                                    continue
                            filled_pos.append(missing_position)
                            filled_pos.append(missing_position + 1)
                            msg[missing_poses[0]] = ngram_ref_pos[0]
                            msg[missing_position + 1] = ngram_ref_pos[0]
                            ref_pos_in = True
                        else:
                            filled_pos.append(possible_msg_pos)
                            msg[possible_msg_pos] = ngram_ref_pos[0]
                            ref_pos_in = True

                # We lack one integer, so fill with 0
                if len(filled_pos) < 3:
                    missing_poses = [x for x in [0, 1, 2] if x not in filled_pos]
                    msg[missing_poses[0]] = 0

                meaning = f"{ref_integer}" f"-{ref_pos}" f"-Pos"
            else:
                self.compo_pos_failures += 1

        # Compositional with positionals, and successfully identified
        if self.compo_pos and found:
            return msg, meaning
        # Compositional but no positionals
        else:
            # Since we have no positional n-grams to work with, we only use integer references
            # This way we choose which integers to describe randomly
            new_obs = self.rng.permuted(
                observation, axis=0
            )  # Not shuffle, as we do not want in-place

            pos = 0
            # Check for infinite loops
            loops_at_pos = 0
            prev_pos = 0
            while True:
                # Message is complete
                if pos == 3:
                    break

                # Infinite loop probably
                if loops_at_pos > 3:
                    # Fill the un-fillable pos with 0
                    integer = 0
                    msg[pos] = integer

                    # R for random
                    reverse_lang_dict["ints"].append("R")
                    reverse_lang_dict["integer_pos"].append("R")
                    reverse_lang_dict["ngram_pos"].append("R")
                    reverse_lang_dict["msg_pos"].append(pos)
                    reverse_lang_dict["ngrams"].append(f"R-{integer}")

                    # Try new pos
                    pos += 1
                    # Reset infinite loop detector
                    loops_at_pos = 0
                    continue

                # Use invariant, or pos?
                if self.rng.random() < 0.5:
                    # Use invariant
                    for idx, integer in enumerate(new_obs):
                        integer = integer.item()
                        if "inv" not in self.language:
                            break
                        if (
                            integer not in filled_ints
                            and integer in self.language["inv"]
                        ):
                            if len(self.language["inv"][integer]) > 1:
                                # Can't do rng choice directly as ngrams have different lengths
                                # And choice tries to convert the list to an array
                                # Therefore raising the non-homogenous array error
                                ngram = self.language["inv"][integer][
                                    self.rng.choice(len(self.language["inv"][integer]))
                                ]
                            else:
                                ngram = self.language["inv"][integer][0]

                            # Ngram cannot go over the length of the message
                            if len(ngram) + pos <= 3:
                                msg[pos : pos + len(ngram)] = ngram
                                filled_ints.append(integer)
                                reverse_lang_dict["ints"].append(integer)
                                reverse_lang_dict["integer_pos"].append(idx)
                                reverse_lang_dict["ngram_pos"].append("inv")
                                reverse_lang_dict["msg_pos"].append(pos)
                                reverse_lang_dict["ngrams"].append(ngram)
                                pos += len(ngram)
                                break
                        else:
                            continue
                else:
                    found = False
                    # Use position variant for given pos
                    for idx, integer in enumerate(new_obs):
                        integer = integer.item()
                        # Sometimes the language doesn't have this position at all.
                        if str(pos) not in self.language:
                            break
                        if (
                            integer not in filled_ints
                            and integer in self.language[str(pos)]
                        ):
                            if len(self.language[str(pos)][integer]) > 1:
                                # Can't do rng choice directly as ngrams have different lengths
                                # And choice tries to convert the list to an array
                                # Therefore raising the non-homogenous array error
                                ngram = self.language[str(pos)][integer][
                                    self.rng.choice(
                                        len(self.language[str(pos)][integer])
                                    )
                                ]
                            else:
                                ngram = self.language[str(pos)][integer][0]

                            # Ngram cannot go over the length of the message
                            if len(ngram) + pos <= 3:
                                msg[pos : pos + len(ngram)] = ngram
                                filled_ints.append(integer)
                                reverse_lang_dict["ints"].append(integer)
                                reverse_lang_dict["integer_pos"].append(idx)
                                reverse_lang_dict["ngram_pos"].append(pos)
                                reverse_lang_dict["msg_pos"].append(pos)
                                reverse_lang_dict["ngrams"].append(ngram)
                                pos += len(ngram)
                                found = True
                                break
                        else:
                            continue

                    if not found:
                        # Use invariant, since position specific was not found
                        for idx, integer in enumerate(new_obs):
                            integer = integer.item()
                            # Sometimes the language doesn't have this position at all.
                            if "inv" not in self.language:
                                break
                            if (
                                integer not in filled_ints
                                and integer in self.language["inv"]
                            ):
                                if len(self.language["inv"][integer]) > 1:
                                    # Can't do rng choice directly as ngrams have different lengths
                                    # And choice tries to convert the list to an array
                                    # Therefore raising the non-homogenous array error
                                    ngram = self.language["inv"][integer][
                                        self.rng.choice(
                                            len(self.language["inv"][integer])
                                        )
                                    ]
                                else:
                                    ngram = self.language["inv"][integer][0]

                                # Ngram cannot go over the length of the message
                                if len(ngram) + pos <= 3:
                                    msg[pos : pos + len(ngram)] = ngram
                                    filled_ints.append(integer)
                                    reverse_lang_dict["ints"].append(integer)
                                    reverse_lang_dict["integer_pos"].append(idx)
                                    reverse_lang_dict["ngram_pos"].append("inv")
                                    reverse_lang_dict["msg_pos"].append(pos)
                                    reverse_lang_dict["ngrams"].append(ngram)
                                    pos += len(ngram)
                                    break
                            else:
                                continue

                if pos != prev_pos:
                    #  Pos has changed
                    prev_pos = pos
                else:
                    # We may be stuck on pos
                    loops_at_pos += 1

            msg = np.array(msg)

            # Add message to reverse language dictionary
            #  we want the meaning without being overwritten
            meaning = (
                f"{reverse_lang_dict['ints']}"
                f"-{reverse_lang_dict['integer_pos']}"
                f"-{reverse_lang_dict['ngram_pos']}"
                f"-{reverse_lang_dict['msg_pos']}"
                f"-{reverse_lang_dict['ngrams']}"
                f"-NonPos"
            )

        return msg, meaning

    def map_message_meaning(self, messages: np.ndarray | torch.Tensor) -> list | str:
        """
        Map an observation to a message.

        Parameters
        ----------
        messages: np.ndarray
            A 1-D or 2-D array of message(s) to be mapped to meanings.
        """
        if isinstance(messages, torch.Tensor):
            if messages.get_device() != -1:
                messages = messages.cpu()
            messages = messages.numpy()

        meanings = []
        if messages.ndim == 1:
            if str(messages).replace(",", "") in self.reverse_lang:
                meanings.append(self.reverse_lang[str(messages).replace(",", "")])
            else:
                meanings.append("Unknown message.")
        elif messages.ndim == 2:
            for message in messages:
                if str(message).replace(",", "") in self.reverse_lang:
                    meanings.append(self.reverse_lang[str(message).replace(",", "")])
                else:
                    meanings.append("Unknown message.")
        else:
            raise ValueError("Cannot accept 3D messages!")

        return meanings

    def _compo_pos_helper(self, observation: np.ndarray | torch.Tensor):
        all_pos = ["npmi_pos_0", "npmi_pos_1", "npmi_pos_2"]
        # prefer more precise deixis
        # so only pick l or r if others not available
        relative_pos_integers = {
            "l1": -1,
            "l2": -1,
            "l3": -1,
            "l4": -1,
            "r1": -1,
            "r2": -1,
            "r3": -1,
            "r4": -1,
            "l": [],
            "r": [],
        }
        target_id = np.where(observation == -1)[0][0]
        for idx in range(1, 5):
            # Traverse left
            if target_id - idx >= 0:
                relative_pos_integers[f"l{idx}"] = observation[target_id - idx].item()
                relative_pos_integers["l"].append(observation[target_id - idx].item())
            # Traverse right
            if target_id + idx < len(observation):
                relative_pos_integers[f"r{idx}"] = observation[target_id + idx].item()
                relative_pos_integers["r"].append(observation[target_id + idx].item())

        valid_ref_pos = [
            k
            for k in relative_pos_integers.keys()
            if relative_pos_integers[k] != -1 and relative_pos_integers[k] != []
        ]

        # Prune empty spots
        relative_pos_integers = {
            k: v for k, v in relative_pos_integers.items() if k in valid_ref_pos
        }

        avail_ref_pos_msg_pos = list(self.compo_message_dict_pos.keys())
        avail_int_msg_pos = list(self.compo_message_dict_int.keys())
        self.rng.shuffle(avail_ref_pos_msg_pos)
        self.rng.shuffle(avail_int_msg_pos)

        # First choose which position to describe
        for referent_pos_msg in avail_ref_pos_msg_pos:
            # Find valid positions for integers
            if referent_pos_msg == "inv_npmi":
                valid_int_poses = all_pos
            else:
                valid_int_poses = [item for item in all_pos if item != referent_pos_msg]

            referent_position_keys = list(
                self.compo_message_dict_pos[referent_pos_msg].keys()
            )
            self.rng.shuffle(referent_position_keys)

            # Random order of which referent positions to check.
            for referent_position_pos in referent_position_keys:
                if referent_position_pos not in valid_ref_pos:
                    continue

                # Valid referent position found
                # Now need to check two things
                # One - do any integers fit this?
                # Two - do any integers fit the message?

                # Go through all possible ngrams randomly
                ngrams = self.compo_message_dict_pos[referent_pos_msg][
                    referent_position_pos
                ]
                self.rng.shuffle(ngrams)
                # Find length of positional reference,
                # since we will need to prune valid positions for the integers
                for ngram_ref_pos in ngrams:
                    max_len = 2
                    if len(ngram_ref_pos) == 2:
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
                    for integer_pos_msg in avail_int_msg_pos:
                        # Can only be in valid poses or invariant, then we can (maybe) find a spot
                        if (
                            integer_pos_msg not in valid_int_poses
                            and integer_pos_msg != "inv_npmi"
                        ):
                            continue
                        # Randomly pick an integer as well
                        possible_integers = list(
                            self.compo_message_dict_int[integer_pos_msg].keys()
                        )
                        if referent_position_pos in ["l", "r"]:
                            possible_integers = [
                                integer
                                for integer in possible_integers
                                if integer
                                in relative_pos_integers[referent_position_pos]
                            ]
                        else:
                            possible_integers = [
                                integer
                                for integer in possible_integers
                                if integer
                                == relative_pos_integers[referent_position_pos]
                            ]
                        if len(possible_integers) > 0:
                            self.rng.shuffle(possible_integers)
                            for possible_integer in possible_integers:
                                ngram_int = self.compo_message_dict_int[
                                    integer_pos_msg
                                ][possible_integer]
                                if len(ngram_int) > 1:
                                    # Multiple possible ngrams for this int
                                    # First shuffle so we don't always pick the same ngram
                                    self.rng.shuffle(ngram_int)
                                    for ngram in ngram_int:
                                        ngram_len = len(ngram)
                                        # Need to see if next position is also available
                                        if ngram_len == 2:
                                            match integer_pos_msg:
                                                case "npmi_pos_0":
                                                    if (
                                                        "npmi_pos_1"
                                                        not in valid_int_poses
                                                    ):
                                                        continue
                                                case "npmi_pos_1":
                                                    if (
                                                        "npmi_pos_2"
                                                        not in valid_int_poses
                                                    ):
                                                        continue
                                                case "npmi_pos_2":
                                                    raise ValueError(
                                                        "n-gram of len 2 cannot be in pos 2"
                                                    )
                                        if ngram_len <= max_len:
                                            return True, [
                                                referent_pos_msg,
                                                ngram_ref_pos,
                                                integer_pos_msg,
                                                ngram,
                                                possible_integer,
                                                referent_position_pos,
                                            ]
                                else:
                                    # Only one ngram here
                                    ngram = ngram_int[0]
                                    ngram_len = len(ngram)
                                    # Need to see if next position is also available
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
                                            referent_pos_msg,
                                            ngram_ref_pos,
                                            integer_pos_msg,
                                            ngram,
                                            possible_integer,
                                            referent_position_pos,
                                        ]
        return False, []
