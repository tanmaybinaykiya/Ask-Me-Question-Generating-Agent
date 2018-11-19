import json

from torch.utils import data
from torch.utils.data import DataLoader


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
        print("GET ITEM: ", self.questionAnswerPairs[index])
        return self.questionAnswerPairs[index]


if __name__ == '__main__':
    val = SquadDataset()
    print(val.__getitem__(1))

    train_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=1)

    for z in next(iter(train_loader)):
        # z = next(iter())
        print("z: ", z)


class CustomDataLoader(DataLoader):

    pass