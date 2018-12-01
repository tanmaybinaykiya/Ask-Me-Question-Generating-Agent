import json
import os
from collections import Counter

import numpy as np

from constants import *


class SquadPreProcessor:

    def __init__(self, path, split, q_vocab_size: int=float("inf"), a_vocab_size: int=float("inf"),
                 paragraphs_path: str = None, question_answer_pairs_path: str = None,
                 q_word_idx_map=None, a_word_idx_map=None, q_idx_word_map=None, a_idx_word_map=None,
                 q_word_to_idx_path: str = None, q_idx_to_word_path: str = None, a_word_to_idx_path: str = None,
                 a_idx_to_word_path: str = None):

        self.dataset_path = path
        assert os.path.isfile(self.dataset_path), "Dataset file [%s] doesn't exist" % self.dataset_path

        self.split = split

        self.paragraphs_path = paragraphs_path if paragraphs_path else DatasetPaths["paragraphs-path"] % self.split
        self.qa_pairs_path = question_answer_pairs_path if question_answer_pairs_path else DatasetPaths[
                                                                                               "question-answer-pairs-path"] % self.split
        self.q_word_to_idx_path = q_word_to_idx_path if q_word_to_idx_path else DatasetPaths["word-to-idx-path"][
                                                                                    "question"] % self.split
        self.q_idx_to_word_path = q_idx_to_word_path if q_idx_to_word_path else DatasetPaths["idx-to-word-path"][
                                                                                    "question"] % self.split
        self.a_word_to_idx_path = a_word_to_idx_path if a_word_to_idx_path else DatasetPaths["word-to-idx-path"][
                                                                                    "answer"] % self.split
        self.a_idx_to_word_path = a_idx_to_word_path if a_idx_to_word_path else DatasetPaths["idx-to-word-path"][
                                                                                    "answer"] % self.split

        if not os.path.isdir("./data/%s" % self.split):
            os.makedirs("./data/%s" % self.split, exist_ok=True)

        if q_word_idx_map and a_word_idx_map and q_idx_word_map and a_idx_word_map:
            self.q_word_to_idx = q_word_idx_map.copy()
            self.q_idx_to_word = q_idx_word_map.copy()
            self.a_word_to_idx = a_word_idx_map.copy()
            self.a_idx_to_word = a_idx_word_map.copy()
            self.compute_idx_word_map = False
        else:
            self.q_word_to_idx = {UNKNOWN: 0, START_TOKEN: 1, END_TOKEN: 2}
            self.q_idx_to_word = {0: UNKNOWN, 1: START_TOKEN, 2: END_TOKEN}
            self.a_word_to_idx = {UNKNOWN: 0, START_TOKEN: 1, END_TOKEN: 2}
            self.a_idx_to_word = {0: UNKNOWN, 1: START_TOKEN, 2: END_TOKEN}
            self.compute_idx_word_map = True

        self.q_vocab: Counter = Counter()
        self.a_vocab: Counter = Counter()
        self.q_vocab_size = q_vocab_size
        self.a_vocab_size = a_vocab_size

    @staticmethod
    def create_small_dataset(left: int = 5, right: int = 10, filename: str = "dataset/squad-train-v1.1.json",
                             pruned_dataset_filename: str = "dataset/squad-train-v1.1-smaller.json"):
        file = open(filename)
        z = json.load(file)
        z["data"] = z["data"][left:right]

        with open(pruned_dataset_filename, "w") as f:
            f.write(json.dumps(z))

    @staticmethod
    def preproc_sentence(sentence):
        curr = [token.lower().strip(" .,?") for token in sentence.split(" ")]
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

        if self.compute_idx_word_map:
            for datum_id, datum in enumerate(data):
                for para_id, para in enumerate(datum["paragraphs"]):
                    periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
                    for qa in para["qas"]:
                        q_s = SquadPreProcessor.preproc_sentence(qa['question'])
                        a_s = SquadPreProcessor.preproc_sentence(
                            (SquadPreProcessor.get_sentence(para["context"], periods,
                                                            qa["answers"][0]["answer_start"])))
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
                        q_s = SquadPreProcessor.preproc_sentence(qa['question'])
                        a_s = SquadPreProcessor.preproc_sentence(
                            (SquadPreProcessor.get_sentence(para["context"], periods,
                                                            qa["answers"][0]["answer_start"])))

                        self.update_word_idx_map(q_s, q=True)
                        self.update_word_idx_map(a_s, q=False)

                        q = self.transform_to_idx(q_s, q=True)
                        a = self.transform_to_idx(a_s, q=False)
                        question_answer_pairs.append((q, a, dict_para_id))
        else:
            for datum_id, datum in enumerate(data):
                for para_id, para in enumerate(datum["paragraphs"]):
                    dict_para_id = datum_id * 1000 + para_id
                    paragraphs[dict_para_id] = para["context"]
                    periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
                    for qa in para["qas"]:
                        q_s = SquadPreProcessor.preproc_sentence(qa['question'])
                        a_s = SquadPreProcessor.preproc_sentence((SquadPreProcessor.get_sentence(para["context"],
                                                                         periods, qa["answers"][0]["answer_start"])))
                        q = self.transform_to_idx(q_s, q=True)
                        a = self.transform_to_idx(a_s, q=False)
                        question_answer_pairs.append((q, a, dict_para_id))

        return paragraphs, question_answer_pairs

    def persist(self, paragraphs, q_a_pairs):
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


class GlovePreproccesor:

    @staticmethod
    def obtain_glove_embeddings(glove_filename, word_to_ix, pruned_glove_filename, overwrite=True):
        assert os.path.isfile(glove_filename), "Glove File doesn't exist"
        if os.path.isfile(pruned_glove_filename) and not overwrite:
            print("%s exists. Loading..." % pruned_glove_filename)
            word_embeddings = np.load(pruned_glove_filename)
        else:
            print("%s doesn't exist. Pruning..." % pruned_glove_filename)
            word_embeddings = GlovePreproccesor.prune_glove_embeddings(glove_filename, word_to_ix)
            np.save(pruned_glove_filename, word_embeddings)
        return word_embeddings

    @staticmethod
    def prune_glove_embeddings(filename, word_to_ix):
        vocab = list(word_to_ix.keys())
        UNK_VECTOR_REPRESENTATION = np.array([0.0] * 300)
        word_vecs = {UNKNOWN: UNK_VECTOR_REPRESENTATION}

        f = open(filename, encoding='utf-8')
        for line in f:
            try:
                values = line.split()
                word = values[0]
                if word in word_to_ix:
                    word_vecs[word] = np.array(values[1:], dtype='float32')
            except ValueError as e:
                print("Error occurred but ignored ", e)

        word_embeddings = []

        for word in vocab:
            if word in word_vecs:
                embed = word_vecs[word]
            else:
                embed = UNK_VECTOR_REPRESENTATION
            word_embeddings.append(embed)

        word_embeddings = np.array(word_embeddings)
        return word_embeddings


def main():
    SquadPreProcessor.create_small_dataset(left=10, right=150)
    train_ds = SquadPreProcessor(path=DatasetPaths["squad"]["small_train"], split="train", q_vocab_size=6000,
                              a_vocab_size=12000)

    paragraphs, question_answer_pairs = train_ds.preprocess()
    train_ds.persist(paragraphs, question_answer_pairs)

    dev = SquadPreProcessor(path=DatasetPaths["squad"]["dev"], split="dev",
                            q_word_idx_map=train_ds.q_word_to_idx, a_word_idx_map=train_ds.a_word_to_idx,
                            q_idx_word_map=train_ds.q_idx_to_word, a_idx_word_map=train_ds.a_idx_to_word)
    paragraphs, question_answer_pairs = dev.preprocess()
    dev.persist(paragraphs, question_answer_pairs)

    GlovePreproccesor().obtain_glove_embeddings(glove_filename=DatasetPaths["glove"]["original-embeddings"],
                                                word_to_ix=train_ds.a_word_to_idx,
                                                pruned_glove_filename=DatasetPaths["glove"]["answer-embeddings-small"])

    GlovePreproccesor().obtain_glove_embeddings(glove_filename=DatasetPaths["glove"]["original-embeddings"],
                                                word_to_ix=train_ds.q_word_to_idx,
                                                pruned_glove_filename=DatasetPaths["glove"][
                                                    "question-embeddings-small"])


if __name__ == '__main__':
    main()
