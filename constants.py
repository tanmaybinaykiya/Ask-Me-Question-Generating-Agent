START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNKNOWN = '<unk>'

DatasetPaths = {
    "paragraphs_path": './data/%s/paragraphs.json',
    "question_answer_pairs_path": "./data/%s/q_a_pairs.json",
    "glove": {
        "original-embeddings": "data/glove.840B.300d.txt",
        "answer-embeddings": "data/answer_glove_embeddings.npy",
        "question-embeddings": "data/question_glove_embeddings.npy",
    },
    "squad": {
        "dev": "dataset/squad-dev-v1.1.json",
        "train": "dataset/squad-train-v1.1.json",
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
