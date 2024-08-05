import math
import random
import warnings
import numpy as np
from functools import cache

random.seed(42)


class Noising_Instruction:
    def __init__(
        self,
        instr: str,
        noise_type: str | None = None,
        noise_level: float | None = None,
        opposite_dict_path: str | None = None,
        trunc_len_mode: str = "fixed",
        replace_token_mode: str = "top",
        randpad_token_mode: str = "top",
    ) -> None:
        self.instr = instr
        self.instr_words = self.instr.split(" ")
        self.instr_len = len(self.instr_words)
        self.noise_type = noise_type
        self.noise_level = noise_level

        self.trunc_len_mode = trunc_len_mode
        self.replace_token_mode = replace_token_mode
        self.randpad_token_mode = randpad_token_mode

        self.processed_instr = None
        self.hard_ratio = None
        self.soft_ratio = None

        self.opposite_dict = opposite_dict_path  # read

        if self.noise_level != None and self.noise_type != None:
            self.process()
            self.hard_eval()

    def process(self) -> None:
        if self.noise_type == None or self.noise_level == None:
            warnings.warn("noise_type and noise_level is None!")
        else:
            getattr(self, self.noise_type)()
            assert self.instr != self.processed_instr, "process function error!"

    def noise(
        self, noise_type: str | None = None, noise_level: float | None = None
    ) -> None:
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.process()

    @property
    @cache
    def noise_len(self) -> int:
        if self.trunc_len_mode == "fixed":
            return math.ceil(self.noise_level * self.instr_len)  # 随机固定trunc
        elif self.trunc_len_mode == "random":
            return random.randint(
                1, math.ceil(self.noise_level * self.instr_len)
            )  # 范围trunc

    def trunc(self) -> None:
        trunc_direction = random.randint(-1, 0)
        if trunc_direction == -1:
            self.processed_instr = " ".join(self.instr_words[: -self.noise_len])
        elif trunc_direction == 0:
            self.processed_instr = " ".join(self.instr_words[self.noise_len :])

        # # debug line
        # print(
        #     f"{self.instr}\n{self.processed_instr}\n{trunc_direction}\n{self.trunc_len}"
        # )

    def trunc_shuffle(self) -> None:
        tmp_words = self.instr_words
        trunc_direction = random.randint(-1, 0)

        if trunc_direction == -1:
            insert_words = self.instr_words[-self.noise_len :]
            tmp_words = self.instr_words[: -self.noise_len]
        elif trunc_direction == 0:
            insert_words = self.instr_words[: self.noise_len]
            tmp_words = self.instr_words[self.noise_len :]

        insert_index = random.randint(1, len(tmp_words) - 1)
        self.processed_instr = " ".join(
            tmp_words[:insert_index] + insert_words + tmp_words[insert_index:]
        )

        ## debug line
        # print(
        #     f"{self.instr}\n{self.processed_instr}\n{trunc_direction}\n{self.trunc_len}"
        # )

    @cache
    def replace(self) -> None:
        if self.replace_token_mode == "top":
            from top_words import TOP3000_WORDS

            self.token_list = TOP3000_WORDS

        replace_index = random.sample(list(range(self.instr_len)), self.noise_len)
        replace_words = random.sample(self.token_list, self.noise_len)

        self.processed_instr = self.instr_words
        for i, j in zip(replace_index, replace_words):
            self.processed_instr[i] = j
        self.processed_instr = " ".join(self.processed_instr)

    def opposite(self) -> None:
        self.processed_instr = self.opposite_dict[self.instr]

    def icl_pad(self) -> None:
        pass

    def random_pad(self) -> None:
        if self.randpad_token_mode == "top":
            from top_words import TOP3000_WORDS

            self.token_list = TOP3000_WORDS

        pad_words = random.sample(self.token_list, self.noise_len)
        split_idx = random.randint(0, len(pad_words) - 1)
        pad_front = " ".join(pad_words[:split_idx]) + "."
        pad_back = " ".join(pad_words[split_idx:]) + "."

        self.processed_instr = pad_front + self.instr + pad_back

    def edit_distance_slow(self) -> int:
        # levenshtein_distance with standard implementation
        processed_words = self.processed_instr.split(" ")

        raw_len = self.instr_len + 1
        new_len = len(processed_words) + 1

        # initialize matrix
        matrix = np.zeros((raw_len, new_len))
        matrix[0] = list(range(new_len))
        matrix[:, 0] = list(range(raw_len))

        # calculate
        for i in range(1, raw_len):
            for j in range(1, new_len):
                cost = 0 if self.instr_words[i - 1] == processed_words[j - 1] else 1
                matrix[i, j] = min(
                    matrix[i - 1, j] + 1,
                    matrix[i, j - 1] + 1,
                    matrix[i - 1, j - 1] + cost,
                )

        return int(matrix[-1, -1])

    def edit_distance_fast(self) -> int:
        # levenshtein_distance with dfs optimization on speed
        processed_words = self.processed_instr.split(" ")

        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return abs(i - j)
            if self.instr_words[i] == processed_words[j]:
                return dfs(i - 1, j - 1)
            return min(dfs(i - 1, j), dfs(i, j - 1), dfs(i - 1, j - 1)) + 1

        return dfs(len(self.instr_words) - 1, len(processed_words) - 1)

    def hard_eval(self) -> None:
        self.hard_ratio = self.edit_distance_fast()

    def soft_eval(self) -> None:
        pass