import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os, time, sys, datetime, argparse, pickle, json

from data_loader import SquadDataset
from torch.utils.data import DataLoader

from models import obtain_glove_embeddings, EncoderBILSTM, DecoderLSTM



val = SquadDataset()

train_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=1)

#for z in next(iter(train_loader)):
    #z = next(iter())
    #print("z: ", z)

num_epoch = 1
batch_size = 1
n_train = len(train_loader)

batch_per_epoch = n_train // batch_size
n_iters = batch_per_epoch * num_epoch


#filename_glove = 'data/glove.840B.300d.txt'
#word_embeddings_glove = obtain_glove_embeddings(filename_glove, train_loader.dataset.word_to_idx)

train_vocab_size = len(train_loader.dataset.word_to_idx)
train_iter = iter(train_loader)

#for qa, (question, answer, pID) in enumerate(train_loader):
    #print("QA", qa, "ANSWER LEN", len(answer))



encoder = EncoderBILSTM(vocab_size=train_vocab_size, n_layers=1, embedding_dim=300, hidden_dim=600, dropout=0) #embeddings=word_embeddings_glove)
decoder = DecoderLSTM(vocab_size=train_vocab_size, embedding_dim=300, hidden_dim=600, n_layers=1, encoder_hidden_dim=600)

criterion = nn.NLLLoss()

for eachEpoch in range(num_epoch):
    for eachBatch in range(batch_per_epoch):
        i = batch_per_epoch * eachEpoch + eachBatch + 1  # global step
        batch = next(train_iter)
        #each batch is size 1 for now

        question, answer, pID = batch

        answer = torch.stack(answer)
        answer = torch.autograd.Variable(answer)

        question = torch.stack(question)
        question = torch.autograd.Variable(question)

        encoder_input, encoder_len = answer, len(answer)
        decoder_input, decoder_len = question, len(question)


        encoder_out, encoder_hidden = encoder(encoder_input, encoder_len)
        decoder_out, decoder_hidden = decoder(decoder_input[:-1, :], encoder_hidden, encoder_out,
                                                   encoder_len)


        break