START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNKNOWN = '<unk>'

DatasetPaths = {
    "paragraphs-path": './data/%s/paragraphs.json',
    "question-answer-pairs-path": "./data/%s/q_a_pairs.json",
    "glove": {
        "original-embeddings": "data/glove.840B.300d.txt",
        "answer-embeddings": "data/answer_glove_embeddings.npy",
        "question-embeddings": "data/question_glove_embeddings.npy",
        "answer-embeddings-small": "data/answer_glove_embeddings_small.npy",
        "question-embeddings-small": "data/question_glove_embeddings_small.npy",
    },
    "squad": {
        "dev": "dataset/squad-dev-v1.1.json",
        "train": "dataset/squad-train-v1.1.json",
        "small_train": "dataset/squad-train-v1.1-smaller.json"
    },
    "word-to-idx-path": {
        "question": "./data/%s/q_word_to_idx.json",
        "answer": "./data/%s/a_word_to_idx.json",
    },
    "idx-to-word-path": {
        "question": "./data/%s/q_idx_to_word.json",
        "answer": "./data/%s/a_idx_to_word.json",
    }
}
