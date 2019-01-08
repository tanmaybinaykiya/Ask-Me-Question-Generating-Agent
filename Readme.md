# Question Generating Agent
  
[Project Webpage]('https://tanmaybinaykiya.github.io/Question-Generating-Agent/')

## Abstract
Our goal is to create an agent that could learn to ask a question provided an informative sentence. Such an agent could be used for creating chatbots, conversational agents, and agents that help in reading comprehension to name a few. We used a Sequence to Sequence Deep Neural Net architecture powered by Global Attention. This architecture allows us to focus on different parts of the input sentence and determine how important each word in the sentence is to the next predicted word. We were successful in overfitting the model on a small dataset, showing us that our model was working as intended, but the difficulty in tuning hyperparameters and learning on the entire dataset was considerable. The ultimate performance we achieved on test set after the model had been trained on the entire dataset was far from ideal. 

## Datasets

### Download Squad 
- [Training Set]('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json') to 'data/squad-train-v1.1.json'
- [Dev Set]('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json') to 'data/squad-dev-v1.1.json'

```sh
mkdir dataset
curl -o 'dataset/squad-train-v1.1.json' 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' 
curl -o 'dataset/squad-dev-v1.1.json' 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' 
```

### Download GloVe

[GloVe 840B 300 dimension vectors]('http://nlp.stanford.edu/data/glove.840B.300d.zip')

```sh
wget 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
unzip 'glove.840B.300d.zip'
rm glove.840B.300d.zip
mv glove.840B.300d.txt data/
```

### Preprocess Data

```sh
python SquadPreProcessor.py
```


