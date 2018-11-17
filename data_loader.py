import json

from torch.utils import data
from torch.utils.data import DataLoader


# QuestionAnswerPair = namedtuple('QuestionAnswer', ['question', 'answer', 'paragraphId'])

class SquadDataset(data.Dataset):
    urls = ['https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json']
    name = 'SquadDataset'
    paragraphs = './data/paragraphs.pt'
    question_answer_pairs_path = "./data/qAPairs.json"
    word_to_idx_path = "./data/word_to_idx.json"
    idx_to_word_path = "./data/idx_to_word.json"

    def __init__(self):
        self.questionAnswerPairs = json.load(self.question_answer_pairs_path)
        self.idx_to_word = json.load(self.word_to_idx_path)
        self.word_to_idx = json.load(self.idx_to_word_path)

    def __len__(self):
        return len(self.questionAnswerPairs)

    def __getitem__(self, index):
        return self.questionAnswerPairs[index]


val = SquadDataset()
print(val.__getitem__(1))
