import json
import os.path
from collections import namedtuple

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from constants import DatasetPaths

# unused:: This is how QuestionAnswerPairs are stored
QuestionAnswerPair = namedtuple('QuestionAnswer', ['question', 'answer', 'paragraphId'])


class SquadDataset(data.Dataset):
    train_url = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json']
    dev_url = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json']
    name = 'SquadDataset'

    def __init__(self, split, paragraphs_path: str = None, question_answer_pairs_path: str = None,
                 q_word_to_idx_path: str = None, q_idx_to_word_path: str = None, a_word_to_idx_path: str = None,
                 a_idx_to_word_path: str = None):
        self.split = split

        self.paragraphs_path = paragraphs_path if paragraphs_path else DatasetPaths["paragraphs-path"] % self.split
        self.question_answer_pairs_path = question_answer_pairs_path if question_answer_pairs_path else DatasetPaths[ "question-answer-pairs-path"] % self.split
        self.q_word_to_idx_path = q_word_to_idx_path if q_word_to_idx_path else DatasetPaths["word-to-idx-path"]["question"] % self.split
        self.q_idx_to_word_path = q_idx_to_word_path if q_idx_to_word_path else DatasetPaths["idx-to-word-path"]["question"] % self.split
        self.a_word_to_idx_path = a_word_to_idx_path if a_word_to_idx_path else DatasetPaths["word-to-idx-path"]["answer"] % self.split
        self.a_idx_to_word_path = a_idx_to_word_path if a_idx_to_word_path else DatasetPaths["idx-to-word-path"]["answer"] % self.split

        assert os.path.isfile(self.paragraphs_path), "Paragraphs file [%s] doesn't exist" % self.paragraphs_path
        assert os.path.isfile(self.question_answer_pairs_path), "qa_pairs file [%s] doesn't exist" % self.question_answer_pairs_path
        assert os.path.isfile(self.q_word_to_idx_path), "q_word_to_idx [%s] file doesn't exist" % self.q_word_to_idx_path
        assert os.path.isfile(self.q_idx_to_word_path), "q_idx_to_word [%s] file doesn't exist" % self.q_idx_to_word_path
        assert os.path.isfile(self.a_word_to_idx_path), "a_word_to_idx [%s] file doesn't exist" % self.a_word_to_idx_path
        assert os.path.isfile(self.a_idx_to_word_path), "a_idx_to_word_path [%s] file doesn't exist" % self.a_idx_to_word_path

        self.paragraphs = json.load(open(self.paragraphs_path, "r"))
        self.questionAnswerPairs = json.load(open(self.question_answer_pairs_path, "r"))
        self.q_idx_to_word = json.load(open(self.q_idx_to_word_path, "r"))
        self.q_word_to_idx = json.load(open(self.q_word_to_idx_path, "r"))
        self.a_idx_to_word = json.load(open(self.a_idx_to_word_path, "r"))
        self.a_word_to_idx = json.load(open(self.a_word_to_idx_path, "r"))

    def __len__(self):
        return len(self.questionAnswerPairs)

    def __getitem__(self, index):
        return self.questionAnswerPairs[index]

    def get_question_idx_to_word(self):
        return self.q_idx_to_word

    def get_question_word_to_idx(self):
        return self.q_word_to_idx

    def get_answer_idx_to_word(self):
        return self.a_idx_to_word

    def get_answer_word_to_idx(self):
        return self.a_word_to_idx

    def get_paragraphs(self):
        return self.paragraphs


def collate_fn(datum):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        datum: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(np.asarray(seq[:end]))
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    datum.sort(key=lambda x: len(x[1]), reverse=True)
    # separate source and target sequences
    src_seqs, trg_seqs, p_id = zip(*datum)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths, p_id


class GloVeEmbeddings:

    @staticmethod
    def load_glove_embeddings(question=True, small=True):
        if small:
            pruned_glove_filename = DatasetPaths["glove"]["question-embeddings-small"] if question else DatasetPaths["glove"]["answer-embeddings-small"]
        else:
            pruned_glove_filename = DatasetPaths["glove"]["question-embeddings"] if question else DatasetPaths["glove"]["answer-embeddings"]
        assert os.path.isfile(pruned_glove_filename), "Glove File[%s] doesn't exist" % pruned_glove_filename
        return np.load(pruned_glove_filename)


def main():
    val = SquadDataset("dev")
    train_loader = DataLoader(val, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for datum in train_loader:
        print(datum[0])


if __name__ == '__main__':
    main()
