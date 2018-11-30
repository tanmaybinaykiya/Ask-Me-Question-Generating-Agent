import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu


def plot_losses(losses):
    plt.plot(losses)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)

    plt.show()


class BleuScorer:

    @staticmethod
    def score(reference, candidate):
        return sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method0)

    @staticmethod
    def corpus_score(references, candidates):
        return corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method0)


def test():
    print(BleuScorer.score(reference=[['this', 'is', 'a', 'test'], ['this', 'is' 'test']],
                           candidate=['is', 'this', 'is', 'a', 'test', 'tanmay']))
    print(BleuScorer.corpus_score(references=[[['this', 'is', 'a', 'test'], ['this', 'is' 'test']],
                                              [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]],
                                  candidates=[['is', 'this', 'is', 'a', 'test', 'tanmay'],
                                              ['this', 'is', 'not', 'a', 'test']]))


if __name__ == '__main__':
    test()
