import json
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch

# QuestionAnswerPair = namedtuple('QuestionAnswer', ['question', 'answer', 'paragraphId'])


class SquadDataset(data.Dataset):
    urls = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json']
    name = 'SquadDataset'
    paragraphs_path = './data/paragraphs.json'
    question_answer_pairs_path = "./data/qAPairs.json"
    word_to_idx_path = "./data/word_to_idx.json"
    idx_to_word_path = "./data/idx_to_word.json"

    def __init__(self):
        self.paragraphs = json.load(open(self.paragraphs_path, "r"))
        self.questionAnswerPairs = json.load(open(self.question_answer_pairs_path, "r"))
        self.idx_to_word = json.load(open(self.word_to_idx_path, "r"))
        self.word_to_idx = json.load(open(self.idx_to_word_path, "r"))

    def __len__(self):
        return len(self.questionAnswerPairs)


    def __getitem__(self, index):
        #print("GET ITEM: ", self.questionAnswerPairs[index])
        return self.questionAnswerPairs[index]

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
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
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs,p_id = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths,p_id


if __name__ == '__main__':
    val = SquadDataset()

    train_loader = DataLoader(val, batch_size=4, shuffle=True, num_workers=0,collate_fn=collate_fn)
    iterator =iter(train_loader)
    for data in train_loader:
        print(data[0])
        #print("z: ", z)


class CustomDataLoader(DataLoader):

    pass