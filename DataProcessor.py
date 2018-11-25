import json
import os
from collections import Counter

from constants import *


class DataProcessor:

    def __init__(self, path, split, vocab_size):
        self.dataset_path = path
        self.split = split
        self.word_to_idx = {UNKNOWN: 0, START_TOKEN: 1, END_TOKEN: 2}
        self.idx_to_word = {0: UNKNOWN, 1: START_TOKEN, 2: END_TOKEN}
        self.vocab = Counter()
        self.vocab_size = vocab_size
        if not os.path.isdir("./data/%s" % self.split):
            os.mkdir("./data/%s" % self.split)
        self.paragraphs_path = "./data/%s/paragraphs.json" % self.split
        self.qa_pairs_path = "./data/%s/q_a_pairs.json" % self.split
        self.word_to_idx_path = "./data/%s/word_to_idx.json" % self.split
        self.idx_to_word_path = "./data/%s/idx_to_word.json" % self.split

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

    def update_word_idx_map(self, words):
        for word in words:
            if word in self.vocab and word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
                self.idx_to_word[self.word_to_idx[word]] = word

    def transform_to_idx(self, words):
        return [self.word_to_idx.get(word, self.word_to_idx[UNKNOWN]) for word in words]

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
                    self.vocab.update(q_s)
                    self.vocab.update(a_s)

        print("Vocab built:", len(self.vocab))
        self.vocab = {el[0]: el[1] for el in self.vocab.most_common(self.vocab_size)}
        print("Vocab pruned:", len(self.vocab))

        for datum_id, datum in enumerate(data):
            for para_id, para in enumerate(datum["paragraphs"]):
                dict_para_id = datum_id * 1000 + para_id
                paragraphs[dict_para_id] = para["context"]
                periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
                for qa in para["qas"]:
                    q_s = DataProcessor.preproc_sentence(qa['question'])
                    a_s = DataProcessor.preproc_sentence(
                        (DataProcessor.get_sentence(para["context"], periods, qa["answers"][0]["answer_start"])))

                    self.update_word_idx_map(q_s)
                    self.update_word_idx_map(a_s)

                    q = self.transform_to_idx(q_s)
                    a = self.transform_to_idx(a_s)
                    question_answer_pairs.append((q, a, dict_para_id))
                print("Datum Id:[%d/%d], Para Id:[%d/%d] completed" % (datum_id, len(data), para_id, len(datum["paragraphs"])))
        return paragraphs, question_answer_pairs

    def preprocess_dataset(self):
        paragraphs, q_a_pairs, = self.preprocess()
        with open(self.paragraphs_path, "w") as f:
            f.write(json.dumps(paragraphs))
        with open(self.qa_pairs_path, "w") as f:
            f.write(json.dumps(q_a_pairs))
        with open(self.word_to_idx_path, "w") as f:
            f.write(json.dumps(self.word_to_idx))
        with open(self.idx_to_word_path, "w") as f:
            f.write(json.dumps(self.idx_to_word))


def main():
    DataProcessor(path="dataset/squad-train-v1.1.json", split="train", vocab_size=45000).preprocess_dataset()
    DataProcessor(path="dataset/squad-dev-v1.1.json", split="dev", vocab_size=28000).preprocess_dataset()


if __name__ == '__main__':
    main()
