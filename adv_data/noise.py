import math
import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass

from corpus import HIGHEST_WORDS, LOWEST_WODDS, RANDOM_WORDS, ICL_DATASET

random.seed(3407)


@dataclass
class AdvInsConfig:
    adv_tokens_len: str = "fixed"  # 'fixed' or 'random'
    gptopposite_data_path: str | None = None


CONFIG = AdvInsConfig()


class AdvIns:
    def __init__(
        self,
        instruction: str,
        adv_method: str | None = None,
        adv_rate: float | None = None,
        config: AdvInsConfig = CONFIG,
    ) -> None:
        self.instruction = instruction
        self.adv_method = adv_method
        self.adv_rate = adv_rate
        self.config = config

        self.instruction_words = self.instruction.split(" ")  # tokenize the instruction
        self.instruction_len = len(self.instruction_words)

        self.adv_instruction = None
        self.hard_value = None
        self.soft_value = None

        if self.adv_rate is not None and self.adv_method is not None:
            self.process()
            self.noise_hard_eval()

    def process(self) -> None:
        getattr(self, self.adv_method)()  # call the function to noise the instruction
        assert (
            self.instruction != self.adv_instruction
        ), "process function result error!"

    def noise(
        self, noise_type: str | None = None, noise_level: float | None = None
    ) -> None:
        self.adv_method = noise_type
        self.adv_rate = noise_level
        self.process()

    @property
    def noise_len(self) -> int:
        if self.config.adv_tokens_len == "fixed":
            return math.ceil(
                self.adv_rate * self.instruction_len
            )  # truncation in a fixed length
        elif self.config.adv_tokens_len == "random":
            return random.randint(
                1, math.ceil(self.adv_rate * self.instruction_len)
            )  # truncation in a range

    def front_truncation(self) -> None:
        self.adv_instruction = " ".join(
            self.instruction_words[self.noise_len :]
        )  # front truncation

    def rear_truncation(self) -> None:
        self.adv_instruction = " ".join(
            self.instruction_words[: -self.noise_len]
        )  # rear truncation

    def random_truncation(self) -> None:
        direction = random.choice([-1, 1])
        if direction == 1:
            self.front_truncation()
        elif direction == -1:
            self.rear_truncation()

    def front_truncation_shuffle(self) -> None:
        insert_words = self.instruction_words[: self.noise_len]
        tmp_words = self.instruction_words[self.noise_len :]
        insert_idx = random.randint(1, len(tmp_words) - 1)

        self.adv_instruction = " ".join(
            tmp_words[:insert_idx] + insert_words + tmp_words[insert_idx:]
        )

    def rear_truncation_shuffle(self) -> None:
        insert_words = self.instruction_words[-self.noise_len :]
        tmp_words = self.instruction_words[: -self.noise_len]
        insert_idx = random.randint(1, len(tmp_words) - 1)
        self.adv_instruction = " ".join(
            tmp_words[:insert_idx] + insert_words + tmp_words[insert_idx:]
        )

    def random_truncation_shuffle(self) -> None:
        direction = random.randint(-1, 1)
        if direction == 1:
            self.front_truncation_shuffle()
        elif direction == -1:
            self.rear_truncation_shuffle()

    def highest_replace(self) -> None:
        replace_idx = random.sample(list(range(self.instruction_len)), self.noise_len)
        replace_words: list[tuple[str, int]] = random.sample(
            HIGHEST_WORDS, self.noise_len
        )
        self.adv_instruction = deepcopy(self.instruction_words)
        for idx, word in zip(replace_idx, replace_words):
            self.adv_instruction[idx] = word[0]
        self.adv_instruction = " ".join(self.adv_instruction)

    def lowest_replace(self) -> None:
        replace_idx = random.sample(list(range(self.instruction_len)), self.noise_len)
        replace_words: list[tuple[str, int]] = random.sample(
            LOWEST_WODDS, self.noise_len
        )
        self.adv_instruction = deepcopy(self.instruction_words)
        for idx, word in zip(replace_idx, replace_words):
            self.adv_instruction[idx] = word[0]
        self.adv_instruction = " ".join(self.adv_instruction)

    def random_replace(self) -> None:
        replace_idx = random.sample(list(range(self.instruction_len)), self.noise_len)
        replace_words: list[int] = random.sample(RANDOM_WORDS, self.noise_len)
        self.adv_instruction = deepcopy(self.instruction_words)
        for idx, word in zip(replace_idx, replace_words):
            self.adv_instruction[idx] = word
        self.adv_instruction = " ".join(self.adv_instruction)

    def synonym_replace(self) -> None:
        pass

    def antonym_repace(self) -> None:
        pass

    def gpt4_opposite(self) -> None:
        pass

    def icl_pad(self) -> None:
        sampled_direction = random.choice([-1, 1])
        sampled_data = random.sample(ICL_DATASET, 10)
        sampled_words = [t for s in sampled_data for t in s.split(" ") if t.isalpha()]
        if sampled_direction == 1:
            sampled_words = sampled_words[: self.noise_len]
        elif sampled_direction == -1:
            sampled_words = sampled_words[-self.noise_len :]
        direction = random.choice([-1, 1])
        if direction == 1:
            self.adv_instruction = " ".join(sampled_words + self.instruction_words)
        elif direction == -1:
            self.adv_instruction = " ".join(self.instruction_words + sampled_words)

    def highest_front_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(HIGHEST_WORDS, self.noise_len)
        self.adv_instruction = " ".join(pad_words + self.instruction_words)

    def highest_rear_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(HIGHEST_WORDS, self.noise_len)
        self.adv_instruction = " ".join(self.instruction_words + pad_words)

    def highest_context_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(HIGHEST_WORDS, self.noise_len)
        insert_idx = random.randint(1, len(pad_words) - 1)
        self.adv_instruction = " ".join(
            pad_words[:insert_idx] + self.instruction_words + pad_words[insert_idx:]
        )

    def lowest_front_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(LOWEST_WODDS, self.noise_len)
        self.adv_instruction = " ".join(pad_words + self.instruction_words)

    def lowest_rear_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(LOWEST_WODDS, self.noise_len)
        self.adv_instruction = " ".join(self.instruction_words + pad_words)

    def lowest_context_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(LOWEST_WODDS, self.noise_len)
        insert_idx = random.randint(1, len(pad_words) - 1)
        self.adv_instruction = " ".join(
            pad_words[:insert_idx] + self.instruction_words + pad_words[insert_idx:]
        )

    def random_context_pad(self) -> None:
        assert self.noise_len > 1, "noise length is too short!"
        pad_words: list[tuple[str, int]] = random.sample(RANDOM_WORDS, self.noise_len)
        insert_idx = random.randint(1, len(pad_words) - 1)
        self.adv_instruction = " ".join(
            pad_words[:insert_idx] + self.instruction_words + pad_words[insert_idx:]
        )

    def edit_distance(self) -> int:
        # levenshtein_distance with standard DP implementation
        processed_words = self.adv_instruction.split(" ")

        raw_len = self.instruction_len + 1
        new_len = len(processed_words) + 1

        # initialize matrix
        matrix = np.zeros((raw_len, new_len))
        matrix[0] = list(range(new_len))
        matrix[:, 0] = list(range(raw_len))

        # calculate
        for i in range(1, raw_len):
            for j in range(1, new_len):
                cost = (
                    0 if self.instruction_words[i - 1] == processed_words[j - 1] else 1
                )
                matrix[i, j] = min(
                    matrix[i - 1, j] + 1,
                    matrix[i, j - 1] + 1,
                    matrix[i - 1, j - 1] + cost,
                )

        return int(matrix[-1, -1])

    def noise_hard_eval(self) -> None:
        self.hard_value = self.edit_distance()

    def noise_soft_eval(self) -> None:
        pass
