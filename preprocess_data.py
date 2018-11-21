import json

from constants import START_TOKEN, END_TOKEN


def trim(token):
    return token.strip(" .,")


def preproc_sentence(sentence):
    # lower_case
    curr = [trim(token.lower()) for token in sentence.split(" ")]
    curr.insert(0, START_TOKEN)
    curr.append(END_TOKEN)
    return curr


def get_sentence(sentences, periodLocs, answer_start):
    if periodLocs:
        if answer_start <= periodLocs[0]:
            return sentences[:periodLocs[0]]
        for idx in range(1, len(periodLocs)):
            if periodLocs[idx - 1] < answer_start <= periodLocs[idx]:
                return sentences[periodLocs[idx - 1]: periodLocs[idx]]
        if answer_start >= periodLocs[-1]:
            return sentences[periodLocs[-1]:]
    else:
        return sentences


def add_to_vocab(words, word_to_idx, idx_to_word):
    for word in words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[word_to_idx[word]] = word


def transform_to_idx(words, word_to_idx):
    return [word_to_idx[word] for word in words]


def preprocess(filename):
    file = open(filename)
    z = json.load(file)
    data = z["data"]

    paragraphs = {}
    questAnswerPairs = []

    word_to_idx = {}
    idx_to_word = {}

    for datum_id, datum in enumerate(data):
        for para_id, para in enumerate(datum["paragraphs"]):
            dict_para_id = datum_id * 1000 + para_id
            paragraphs[dict_para_id] = para["context"]
            periods = [idx for idx, char in enumerate(para["context"]) if char == '.']
            for qa in para["qas"]:
                q_s = preproc_sentence(qa['question'])
                a_s = preproc_sentence((get_sentence(para["context"], periods, qa["answers"][0]["answer_start"])))

                add_to_vocab(q_s, word_to_idx, idx_to_word)
                add_to_vocab(a_s, word_to_idx, idx_to_word)

                q = transform_to_idx(q_s, word_to_idx)
                a = transform_to_idx(a_s, word_to_idx)
                questAnswerPairs.append((q, a, dict_para_id))
    return paragraphs, questAnswerPairs, word_to_idx, idx_to_word


def main():
    paragraphs, qAPairs, word_to_idx, idx_to_word = preprocess("dataset/squad.json")

    with open("./data/paragraphs.json", "x") as f:
        f.write(json.dumps(paragraphs))
    with open("./data/qAPairs.json", "x") as f:
        f.write(json.dumps(qAPairs))
    with open("./data/word_to_idx.json", "x") as f:
        f.write(json.dumps(word_to_idx))
    with open("./data/idx_to_word.json", "x") as f:
        f.write(json.dumps(idx_to_word))


if __name__ == '__main__':
    main()
