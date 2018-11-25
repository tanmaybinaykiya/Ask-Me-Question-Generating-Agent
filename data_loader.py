import json

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


# QuestionAnswerPair = namedtuple('QuestionAnswer', ['question', 'answer', 'paragraphId'])


class SquadDataset(data.Dataset):
    train_url = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json']
    dev_url = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json']
    name = 'SquadDataset'

    paragraphs_path = './data/%s/paragraphs.json'
    question_answer_pairs_path = "./data/%s/qAPairs.json"
    word_to_idx_path = "./data/%s/word_to_idx.json"
    idx_to_word_path = "./data/%s/idx_to_word.json"

    def __init__(self, split):
        self.paragraphs = json.load(open(self.paragraphs_path % split, "r"))
        self.questionAnswerPairs = json.load(open(self.question_answer_pairs_path % split, "r"))
        self.idx_to_word = json.load(open(self.idx_to_word_path % split, "r"))
        self.word_to_idx = json.load(open(self.word_to_idx_path % split, "r"))

    def __len__(self):
        return len(self.questionAnswerPairs)

    def __getitem__(self, index):
        # print("GET: Q:", self.questionAnswerPairs[index][0],
        #       [self.idx_to_word[str(el)] for el in val.__getitem__(index)[0]])
        # print("GET: A:", self.questionAnswerPairs[index][1],
        #       [self.idx_to_word[str(el)] for el in val.__getitem__(index)[1]])
        # print("GET: P:", self.questionAnswerPairs[index][2],
        #       [self.idx_to_word[str(el)] for el in val.__getitem__(index)[2]])
        return self.questionAnswerPairs[index]

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_paragraphs(self):
        return self.paragraphs


def collate_fn(datum):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
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


def main():
    val = SquadDataset("dev")
    train_loader = DataLoader(val, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for datum in train_loader:
        print(datum[0])


if __name__ == '__main__':
    main()
