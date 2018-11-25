# Question Generating Agent

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
