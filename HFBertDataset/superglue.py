from abc import abstractclassmethod
import torch
import json

class SuperGLUE(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    def make_input(self, tokenizer, template, max_encoder_length, label):
        input = tokenizer(
            template, 
            max_length=max_encoder_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        labels = torch.tensor(label, dtype=torch.long)

        self.data.append({
            "input_ids": input['input_ids'][0].cuda(),
            "attention_mask": input['attention_mask'][0].cuda(),
            "labels": labels.cuda(),
        })

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            max_id = (len(lines)) // world_size * world_size
            for i, row in enumerate(lines[:max_id]):
                yield json.loads(row)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BoolQ_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length) -> None:
        super().__init__()

        from tqdm import tqdm
        for row in self.read_data("BoolQ", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row['passage']
            text_b = row['question']

            template = f'{text_a} [SEP] {text_b}'

            self.make_input(tokenizer, template, max_encoder_length, label)


class CB_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("CB", path, split, rank, world_size):
            if row["label"]=="contradiction":
                label = 0
            elif row["label"]=="entailment":
                label = 1
            else:
                label = 2
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2?' # TODO

            self.make_input(tokenizer, template, max_encoder_length, label)


class COPA_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template = f'Choice 1: {choice_1} Choice 2: {choice_2} The {question} of "{text}" was choice'

            self.make_input(tokenizer, template, max_encoder_length, label)

            if split == 'train': # mirror
                label = label ^ 1
                template = f'Choice 1: {choice_2} Choice 2: {choice_1} The {question} of "{text}" was choice'

                self.make_input(tokenizer, template, max_encoder_length, label)

class MultiRC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()
        self.qids = []

        for template, label, qid in self.read_data("MultiRC", path, split, rank, world_size):
            self.make_input(tokenizer, template, max_encoder_length, label)
            self.qids.append(qid)

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            max_id = (len(lines)) // world_size * world_size
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1

        max_id = (cnt) // world_size * world_size
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1
                        if cnt > max_id: break
                        if cnt % world_size != rank: continue
                        answer = answer_json["text"]
                        label = answer_json["label"]

                        template = f'{text} Is answer "{answer}" the answer to the question "{question}"?'

                        qid = f'{row["idx"]}-{question_json["idx"]}'

                        yield (template, label, qid)


class ReCoRD_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("ReCoRD", path, split, rank, world_size):
            label = row["idx"]
            text = row["passage"]["text"]
            
            entities = []
            for entity_json in row['passage']['entities']:
                start = entity_json['start']
                end = entity_json['end']
                entity = text[start:end+1]
                entities.append(entity)

            text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations

            for question_json in row["qas"]:
                question = question_json["query"]
                answers = []
                for answer_json in question_json["answers"]:
                    answer = answer_json["text"]
                    answers.append(answer)

                template = f'{text} Question: {question} Entities: {entities} Which entities can be filled in the placeholder?'

                self.make_input(tokenizer, template, max_encoder_length, label)

class RTE_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("RTE", path, split, rank, world_size):
            label = 0 if row["label"]=="not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2?'

            self.make_input(tokenizer, template, max_encoder_length, label)


class WiC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("WiC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row["sentence1"]
            text_b = row["sentence2"]
            word = row["word"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does the word {word} in sentence 1 express the same meaning as in sentence 2?'

            self.make_input(tokenizer, template, max_encoder_length, label)


class WSC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length):
        super().__init__()

        for row in self.read_data("WSC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text = row["text"]
            
            span_1 = row["target"]["span1_text"]
            span_2 = row["target"]["span2_text"]

            template = f'{text} Does {span_2} refers to {span_1}?'

            self.make_input(tokenizer, template, max_encoder_length, label)
