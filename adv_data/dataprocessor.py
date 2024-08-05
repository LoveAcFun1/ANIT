import os
import json
import random
from tqdm import tqdm
from noise import AdvIns


class DataProcessor:
    def __init__(
        self,
        data_path: str | None = None,
        save_path: str = None,
        json_format: str = "one_line",
    ) -> None:
        self.data_path = data_path
        self.save_path = save_path
        self.json_format = json_format

        if data_path != None:
            self.load_data()

    def load_data(self):
        if self.json_format == "one_line":
            # if self.data_path.split(".")[-1] == "json":
            with open(self.data_path, "r") as f:
                self.data = json.loads(f.read())
        elif self.json_format == "lines":
            with open(self.data_path, "r") as f:
                self.data = [json.loads(i) for i in f.read().splitlines()]

    def input_template(self, task_type: str) -> dict:
        if task_type == "classification":
            return {
                "instruction": "",
                "adv_instruction": "",
                "question": "",
                "format": "",
                "option": "",
                "input": "",
                "output": "",
            }

        if task_type == "regression":
            return {"instruction": "", "input": "", "output": ""}

    def set_datapath(self, data_path: str):
        self.data_path = data_path
        self.load_data()

    @property
    def datasetpath_dict(self) -> dict:
        return {
            "conll03": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/NER/CoNLL2003/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/NER/CoNLL2003/test.json",
            },
            "ontonotes": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/NER/Ontonotes/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/NER/Ontonotes/test.json",
            },
            "bc4chemd": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/NER/bc4chemd/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/NER/bc4chemd/test.json",
            },
            "scierc": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/RE/SciERC/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/RE/SciERC/test.json",
            },
            "ade": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/RE/ADE_corpus/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/RE/ADE_corpus/test.json",
            },
            "nyt": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/RE/New-York-Times-RE/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/RE/New-York-Times-RE/test.json",
            },
            "14lap": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14lap/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14lap/test.json",
            },
            "14res": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14res/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14res/test.json",
            },
            "16res": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/ABSA/16res/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/ABSA/16res/test.json",
            },
            "14lap+14res+16res": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14lapres_16res/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/ABSA/14lapres_16res/train.json",
            },
            "agnews": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/tex_class/ag_news/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/tex_class/ag_news/test.json",
            },
            "sst2": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/tex_class/sst2/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/tex_class/sst2/dev.json",
            },
            "sst5": {
                "train": "/cto_labs/baishengyuan/noise_llm_data/tex_class/sst5/train.json",
                "test": "/cto_labs/baishengyuan/noise_llm_data/tex_class/sst5/test.json",
            },
        }

    def ner_question(self) -> str:
        return "Give a word that describes the category of the entity word."

    def ner_format(self) -> str:
        return "The output format should be 'type1: word1; type2: word2; ......'. Recognize only the category in options."

    def ner_instructions(self, mode: str) -> list[str]:
        if mode == "short":
            return [
                "Here is a task on named entity recognition, give the correct answer followed the input."
            ]
        elif mode == "long":
            return ["."]
        elif mode == "mixed":
            return ["."]

    def re_question(self) -> str:
        return "Give a phrase that describes the relationship between two words."

    def re_format(self) -> str:
        return "The output format should be 'relation1: word1, word2; relation2: word3, word4; ......'. Extract only the relationship in options."

    def re_instructions(self, mode: str) -> list[str]:
        if mode == "short":
            return [
                "Here is a task on relationship extraction, give the correct answer followed the input."
            ]
        elif mode == "long":
            return ["."]
        elif mode == "mixed":
            return ["."]

    def absa_question(self) -> str:
        return "Give a word that describes the emotional polarity relationship between two words."

    def absa_format(self) -> str:
        return "The output format should be 'emotional1: word1, word2; emotional2: word3, word4; ......'. Extract only the emotional polarity relationship in options."

    def absa_instruction(self, mode: str) -> list[str]:
        if mode == "short":
            return [
                "Here is a task on aspect-based sentiment analysis, give the correct answer followed the input."
            ]
        elif mode == "long":
            return ["."]
        elif mode == "mixed":
            return ["."]

    def tc_question(self) -> str:
        return "Give a label that describes the sequence type."

    def tc_format(self) -> str:
        return "The output format should be 'label: type'. The type is only in the options."

    def tc_instruction(self, mode: str) -> list[str]:
        if mode == "short":
            return [
                "Here is a task on text classification, give the correct answer followed the input."
            ]
        elif mode == "long":
            return ["."]
        elif mode == "mixed":
            return ["."]

    def scierc(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: float | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            # "<noise_s>" + AdvIns(instruction, adv_type, adv_rate) + "<nosie_e>"
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.re_question()
            template["format"] = self.re_format()
            template[
                "option"
            ] = "conjunction, feature of, hyponym of, used for, part of, compare, evaluate for."
            template["input"] = sample["sentence"]
            relations = []
            for i in sample["relations"]:
                relations.append(
                    f"{i['type']}: {i['head']['name']}, {i['tail']['name']}"
                )

            template["output"] = "; ".join(relations)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def nyt(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.re_question()
            template["format"] = self.re_format()
            template["option"] = (
                "ethnicity, place lived, geographic distribution, company industry, country of administrative divisions, "
                + "administrative division of country, location contains, person of company, profession, ethnicity of people, "
                + "company shareholder among major shareholders, sports team of location, religion, neighborhood of, "
                + "company major shareholders, place of death, nationality, children, company founders, company founded place, "
                + "country of capital, company advisors, sports team location of teams, place of birth."
            )
            template["input"] = sample["sentence"]
            relations = []
            for i in sample["relations"]:
                relations.append(
                    f"{i['type']}: {i['head']['name']}, {i['tail']['name']}"
                )

            template["output"] = "; ".join(relations)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def ade(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.re_question()
            template["format"] = self.re_format()
            template["option"] = "adverse effect."
            template["input"] = sample["sentence"]
            relations = []
            for i in sample["relations"]:
                relations.append(
                    f"{i['type']}: {i['head']['name']}, {i['tail']['name']}"
                )

            template["output"] = "; ".join(relations)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def conll03(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.ner_question()
            template["format"] = self.ner_format()
            template["option"] = "location, else, organization, person."
            template["input"] = sample["sentence"]
            categories = []
            for i in sample["entities"]:
                categories.append(f"{i['type']}: {i['name']}")

            template["output"] = "; ".join(categories)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def ontonotes(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.ner_question()
            template["format"] = self.ner_format()
            template["option"] = (
                "date, organization, person, geographical social political, national religious political, facility,"
                + "cardinal, location, work of art, law, event, product, ordinal, percent, time, quantity, money, language"
            )
            template["input"] = sample["sentence"]
            categories = []
            for i in sample["entities"]:
                categories.append(f"{i['type']}: {i['name']}")

            template["output"] = "; ".join(categories)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def bc4chemd(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.ner_question()
            template["format"] = self.ner_format()
            template["option"] = "Chemical."
            template["input"] = sample["sentence"]
            categories = []
            for i in sample["entities"]:
                categories.append(f"{i['type']}: {i['name']}")

            template["output"] = "; ".join(categories)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def lapres14_res16(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.absa_question()
            template["format"] = self.absa_format()
            template["option"] = "Positive, Negative, Neutral."
            template["input"] = sample["sentence"]
            relations = []
            for i in sample["relations"]:
                relations.append(
                    f"{i['type']}: {i['head']['name']}, {i['tail']['name']}"
                )

            template["output"] = "; ".join(relations)

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def agnews(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.tc_question()
            template["format"] = self.tc_format()
            template["option"] = "Sci/Tech, Sports, World, Business."
            template["input"] = sample["text"]
            template["output"] = f"label: {sample['label_text']}"

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def sst2(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.tc_question()
            template["format"] = self.tc_format()
            template["option"] = "positive, negative."
            template["input"] = sample["text"]
            template["output"] = f"label: {sample['label_text']}"

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))

    def sst5(
        self,
        instructions: list,
        adv_type: str | None = None,
        adv_rate: str | None = None,
        task_type: str = "classification",
    ) -> None:
        new_data = []
        for sample in tqdm(self.data):
            template = self.input_template(task_type)
            instruction = random.choice(instructions)
            template["instruction"] = instruction
            assert adv_type in dir(AdvIns)
            template["adv_instruction"] = AdvIns(
                instruction, adv_type, adv_rate
            ).adv_instruction
            template["question"] = self.tc_question()
            template["format"] = self.tc_format()
            template[
                "option"
            ] = "positive, negative, neutral, very negative, very positive."
            template["input"] = sample["text"]
            template["output"] = f"label: {sample['label_text']}"

            new_data.append(template)

        with open(self.save_path, "w") as f:
            f.write(json.dumps(new_data))


if __name__ == "__main__":
    pass
