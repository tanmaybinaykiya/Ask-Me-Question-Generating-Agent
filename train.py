import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os, time, sys, datetime, argparse, pickle, json

from data_loader import SquadDataset,collate_fn
from torch.utils.data import DataLoader

from models import obtain_glove_embeddings, EncoderBILSTM, DecoderLSTM



val = SquadDataset()


#for z in next(iter(train_loader)):
    #z = next(iter())
    #print("z: ", z)

num_epoch = 15
batch_size = 64
train_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn,pin_memory=True)

n_train = len(train_loader)
batch_per_epoch = n_train // batch_size
n_iters = batch_per_epoch * num_epoch


#filename_glove = 'data/glove.840B.300d.txt'
#word_embeddings_glove = obtain_glove_embeddings(filename_glove, train_loader.dataset.word_to_idx)

train_vocab_size = len(train_loader.dataset.word_to_idx)
train_iter = iter(train_loader)

#for qa, (question, answer, pID) in enumerate(train_loader):
    #print("QA", qa, "ANSWER LEN", len(answer))

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=8):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

encoder = EncoderBILSTM(vocab_size=train_vocab_size, n_layers=1, embedding_dim=300, hidden_dim=600, dropout=0) #embeddings=word_embeddings_glove)
decoder = DecoderLSTM(vocab_size=train_vocab_size, embedding_dim=300, hidden_dim=600, n_layers=1, encoder_hidden_dim=600)

if torch.cuda.is_available():
    encoder=encoder.cuda()
    decoder=decoder.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_enc=torch.optim.SGD(encoder.parameters(),lr=1.0)
optimizer_dec=torch.optim.SGD(decoder.parameters(),lr=1.0)
total_batch_loss=0
for eachEpoch in range(num_epoch):
    for eachBatch in range(batch_per_epoch):
        i = batch_per_epoch * eachEpoch + eachBatch + 1  # global step
        batch = next(train_iter)
        #each batch is size 1 for now

        questions,questions_org_len,answers,answers_org_len, pID = batch

        if torch.cuda.is_available():
            questions=questions.cuda()
            answers=answers.cuda()

        #answer = torch.stack(answers,answers_org_len)
        #answer = torch.autograd.Variable(answers)


        #question = torch.stack(question)
        #question = torch.autograd.Variable(question)

        encoder_input, encoder_len = answers, answers.shape[1]
        decoder_input, decoder_len = questions, questions.shape[1]



        encoder_out, encoder_hidden = encoder(encoder_input, torch.LongTensor(answers_org_len).cuda())
        decoder_out, decoder_hidden = decoder(decoder_input[:,:-1], encoder_hidden, encoder_out,
                                                   encoder_len)

        decoder_out=decoder_out.transpose(0,1).contiguous()
        mask=torch.zeros(decoder_out.shape)
        if torch.cuda.is_available():
            mask=mask.cuda()
        mask=mask==1
        for i in range(batch_size):
            mask[i,questions_org_len[i]-1:len(mask[i])]=1
        decoder_out.masked_fill_(mask,0)
        decoder_out=decoder_out.transpose(1,2).contiguous()
        loss=criterion(decoder_out,questions[:,:-1])
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
        total_batch_loss+= loss.item()
        break
    print("Loss for the batch is")
    print(total_batch_loss/batch_per_epoch)

torch.save(encoder.state_dict(),"model_weights/encoder.pth")
torch.save(decoder.state_dict(),"model_weights/decoder.pth")




