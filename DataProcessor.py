import json
import os
from collections import Counter

from constants import *


class DataProcessor:

    def __init__(self, path, split, q_vocab_size, a_vocab_size):
        self.dataset_path = path
        self.split = split
        self.q_word_to_idx = {UNKNOWN: 0, START_TOKEN: 1, END_TOKEN: 2}
        self.q_idx_to_word = {0: UNKNOWN, 1: START_TOKEN, 2: END_TOKEN}
        self.a_word_to_idx = {UNKNOWN: 0, START_TOKEN: 1, END_TOKEN: 2}
        self.a_idx_to_word = {0: UNKNOWN, 1: START_TOKEN, 2: END_TOKEN}
        self.q_vocab = Counter()
        self.a_vocab = Counter()
        self.q_vocab_size = q_vocab_size
        self.a_vocab_size = a_vocab_size
        if not os.path.isdir("./data/%s" % self.split):
            os.makedirs("./data/%s" % self.split, exist_ok=True)
        self.paragraphs_path = "./data/%s/paragraphs.json" % self.split
        self.qa_pairs_path = "./data/%s/q_a_pairs.json" % self.split
        self.q_word_to_idx_path = "./data/%s/q_word_to_idx.json" % self.split
        self.q_idx_to_word_path = "./data/%s/q_idx_to_word.json" % self.split
        self.a_word_to_idx_path = "./data/%s/a_word_to_idx.json" % self.split
        self.a_idx_to_word_path = "./data/%s/a_idx_to_word.json" % self.split

    @staticmethod
    def preproc_sentence(sentence):
        curr = [token.lower().strip(" .,") for token in sentence.split(" ")]
        curr.insert(0, START_TOKEN)
        curr.append(END_TOKEN)
        return curr

    @staticmethod
    def get_sentence(sentences, period_locs, answer_start):
        if period_locs:
            if answer_start <= period_locs[0]:
                return sentences[:period_locs[0]]
            for idx in range(1, len(period_locs)):
                if period_locs[idx - 1] < answer_start <= period_locs[idx]:
                    return sentences[period_locs[idx - 1]: period_locs[idx]]
            if answer_start >= period_locs[-1]:
                return sentences[period_locs[-1]:]
        else:
            return sentences

    def update_word_idx_map(self, words, q):
        if q:
            word_to_idx = self.q_word_to_idx
            idx_to_word = self.q_idx_to_word
            vocab = self.q_vocab
        else:
            word_to_idx = self.a_word_to_idx
            idx_to_word = self.a_idx_to_word
            vocab = self.a_vocab
        for word in words:
            if word in vocab and word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                idx_to_word[word_to_idx[word]] = word

    def transform_to_idx(self, words, q):
        if q:
            return [self.q_word_to_idx.get(word, self.q_word_to_idx[UNKNOWN]) for word in words]
        else:
            return [self.a_word_to_idx.get(word, self.a_word_to_idx[UNKNOWN]) for word in words]

    def preprocess(self):
        file = open(self.dataset_path)
        z = json.load(file)
        data = z["data"]

        paragraphs = {}
        question_answer_pairs = []

        for datum_id, datum in enumerate(data):
            for para_id, para in enumerate(datum["paragraphs"]):
                periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
                for qa in para["qas"]:
                    q_s = DataProcessor.preproc_sentence(qa['question'])
                    a_s = DataProcessor.preproc_sentence(
                        (DataProcessor.get_sentence(para["context"], periods, qa["answers"][0]["answer_start"])))
                    self.q_vocab.update(q_s)
                    self.a_vocab.update(a_s)

        self.q_vocab = {el[0]: el[1] for el in self.q_vocab.most_common(self.q_vocab_size)}
        self.a_vocab = {el[0]: el[1] for el in self.a_vocab.most_common(self.a_vocab_size)}

        for datum_id, datum in enumerate(data):
            for para_id, para in enumerate(datum["paragraphs"]):
                dict_para_id = datum_id * 1000 + para_id
                paragraphs[dict_para_id] = para["context"]
                periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
                for qa in para["qas"]:
                    q_s = DataProcessor.preproc_sentence(qa['question'])
                    a_s = DataProcessor.preproc_sentence(
                        (DataProcessor.get_sentence(para["context"], periods, qa["answers"][0]["answer_start"])))

                    self.update_word_idx_map(q_s, q=True)
                    self.update_word_idx_map(a_s, q=False)

                    q = self.transform_to_idx(q_s, q=True)
                    a = self.transform_to_idx(a_s, q=False)
                    question_answer_pairs.append((q, a, dict_para_id))
        return paragraphs, question_answer_pairs

    def preprocess_dataset(self):
        paragraphs, q_a_pairs, = self.preprocess()
        with open(self.paragraphs_path, "w") as f:
            f.write(json.dumps(paragraphs))
        with open(self.qa_pairs_path, "w") as f:
            f.write(json.dumps(q_a_pairs))
        with open(self.q_word_to_idx_path, "w") as f:
            f.write(json.dumps(self.q_word_to_idx))
        with open(self.q_idx_to_word_path, "w") as f:
            f.write(json.dumps(self.q_idx_to_word))
        with open(self.a_word_to_idx_path, "w") as f:
            f.write(json.dumps(self.a_word_to_idx))
        with open(self.a_idx_to_word_path, "w") as f:
            f.write(json.dumps(self.a_idx_to_word))


def main():
    DataProcessor(path="dataset/squad-train-v1.1.json", split="train", q_vocab_size=45000, a_vocab_size=28000).preprocess_dataset()
    DataProcessor(path="dataset/squad-dev-v1.1.json", split="dev", q_vocab_size=45000, a_vocab_size=28000).preprocess_dataset()


if __name__ == '__main__':
    main()
