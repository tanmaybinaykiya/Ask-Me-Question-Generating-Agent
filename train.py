import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from DataLoader import GloVeEmbeddings, SquadDataset, collate_fn
from constants import *
from evaluation import *
from models import DecoderLSTM, EncoderBILSTM
torch.random.manual_seed(3435)


def exp_lr_scheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=8):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch < lr_decay_epoch:
        return optimizer
    print(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def greedy_search(encoder: EncoderBILSTM, decoder: DecoderLSTM, dataset: torch.utils.data.Dataset, use_cuda: bool, batch_size: int) -> (list, list, list):
    q_idx_to_word = dataset.get_question_idx_to_word()
    a_idx_to_word = dataset.get_answer_idx_to_word()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    given_sentence=[]
    ground_truth=[]
    final_prediction=[]
    encoder.eval()
    decoder.eval()
    max_len = 30
    #batch = iter(data_loader).next()

    for cnt,batch in enumerate(data_loader):
        print(cnt)
        if cnt>200:
            break
        questions, questions_org_len, answers, answers_org_len, pID = batch

        if use_cuda:
            questions = questions.cuda()
            answers = answers.cuda()
            answers_org_len = torch.FloatTensor(np.asarray(answers_org_len))

        attn = torch.zeros(max_len, answers.shape[1])

        encoder_input, encoder_len = answers, np.asarray(answers_org_len)

        if use_cuda:
            encoder_len = torch.LongTensor(encoder_len).cuda()
            decoder_inp = torch.ones((batch_size, 1), dtype=torch.long).cuda()
        else:
            encoder_len = torch.LongTensor(encoder_len)
            decoder_inp = torch.ones((batch_size, 1), dtype=torch.long)
        encoder_out, encoder_hidden = encoder(encoder_input, encoder_len)
        decoder_hidden = encoder_hidden
        # input to the first time step of decoder is <SOS> token.

        seq_len = 0
        eval_mode = False
        predicted_sequences = []
        while seq_len < max_len:
            seq_len += 1
            decoder_out, decoder_hidden,attn_scores = decoder(decoder_inp, decoder_hidden, encoder_out, answers_org_len, eval_mode=eval_mode)

            #attn[seq_len - 1, :] += attn_scores.squeeze().cpu().data
            # obtaining log_softmax scores we need to minimize log softmax over a span.
            decoder_out = decoder_out.view(batch_size, -1)
            decoder_out = torch.nn.functional.log_softmax(decoder_out, )
            prediction = torch.argmax(decoder_out, 1).unsqueeze(1)
            predicted_sequences.append(prediction)
            decoder_inp = prediction.clone()
            eval_mode = True

        given_sentence.extend([[a_idx_to_word[str(answers[i][j].item())] for j in range(len(answers[i])) if answers[i][j] != 0] for i in range(len(answers))])
        ground_truth.extend([[q_idx_to_word[str(questions[i][j].item())] for j in range(len(questions[i])) if questions[i][j] != 0] for i in range(len(questions))])
        prediction=[]
        for i in range(batch_size):
            prediction.append([])
            for j in range(len(predicted_sequences)):
                if q_idx_to_word[str(predicted_sequences[j][i][0].item())] == END_TOKEN:
                    prediction[i].append(END_TOKEN)
                    break
                prediction[i].append(q_idx_to_word[str(predicted_sequences[j][i][0].item())])
        #show_attention(given_sentence,prediction,attn)
        final_prediction.extend(prediction)

    cnt=0
    for sent, gt, pred in zip(given_sentence, ground_truth, final_prediction):
        if cnt<1000:
            cnt+=1
            print("Sentence: %s \nGT Q: %s \nPred Q: %s" % (sent, gt, pred))
        else:
            break
    return [given_sentence], ground_truth, final_prediction


def beam_search(encoder, decoder, dev_loader, dev_idx_to_word_q):
    batch_size = 2
    beam_span = 5
    # encoder.load_state_dict(torch.load("model_weights/encoder.pth"))
    # decoder.load_state_dict(torch.load("model_weights/decoder.pth"))
    encoder.eval()
    decoder.eval()
    for batch in dev_loader:
        questions, questions_org_len, answers, answers_org_len, pID = batch

        if torch.cuda.is_available():
            questions = questions.cuda()
            answers = answers.cuda()

        encoder_input, encoder_len = answers, np.asarray(answers_org_len)
        decoder_input, decoder_len = questions, questions.shape[1]
        encoder_len = torch.LongTensor(encoder_len)
        if torch.cuda.is_available():
            encoder_len = torch.LongTensor(encoder_len).cuda()
        encoder_out, encoder_hidden = encoder(encoder_input, torch.LongTensor(encoder_len))
        decoder_hidden = encoder_hidden

        # input to the first time step of decoder is <SOS> token.
        decoder_inp = torch.zeros((batch_size, 1), dtype=torch.long)
        final_scores = torch.zeros(batch_size, 25)
        final_indices = torch.zeros((batch_size, 25), dtype=torch.long)
        if torch.cuda.is_available():
            decoder_inp = decoder_inp.cuda()
            final_scores = final_indices.cuda()
            final_indices = final_indices.cuda()

        eval_mode = False
        parent_word = []
        actual_indices = []

        # list of all hidden states for a batch
        h_filler = []
        # list of all cell states for a batch
        c_filler = []

        # iterating through every word in the batch
        for word_index in range(decoder_len):
            # iterating through every beam_span proposal for ever word in the batch len(decoder_inp[0]) is the number of words each batch proposes.
            for j in range(len(decoder_inp[0])):
                # if the word is not the start word we have to pass in what the hidden and cell states for the LSTM would be
                if word_index > 0:
                    # find the hidden state and cell for the corresponding word proposed in the beam
                    h1 = [decoder_hidden_per_word[k][j] for k in range(batch_size)]
                    c1 = [cell_per_word[k][j] for k in range(batch_size)]

                    # reshape it to (N,B,Dim)
                    decoder_hidden = torch.stack(h1, 0).unsqueeze(0)
                    cell = torch.stack(c1, 0).unsqueeze(0)

                    # merge the hidden and state cells into a tuple
                    decoder_hidden = (decoder_hidden, cell)

                # passing the jth word predicted by the decoder in the previous step as input.
                decoder_out, dh = decoder(decoder_inp[:, j].view(-1, 1), decoder_hidden, encoder_out, torch.FloatTensor(answers_org_len), eval_mode=eval_mode)

                # obtaining log_softmax scores we need to minimize log softmax over a span.
                decoder_out = decoder_out.view(batch_size, -1)
                decoder_out = torch.nn.functional.log_softmax(decoder_out)

                # Minimizing log likelihood is equal to choosing the max of - log likelihood
                scores, indices = torch.topk(-decoder_out, k=beam_span, )

                # we need to save scores for all spans of the beam if beam span is 5 then every word in the batch can predict 5 words the next two arrays keep track
                # of scores and the indices, adding log likelihood scores is equivalent to multiplying probs.
                final_scores[:, j * beam_span:j * beam_span + beam_span] += scores
                final_indices[:, j * beam_span:j * beam_span + beam_span] = indices

                # we need to keep track of the hidden and cell states for each predicted word since they will be needed in the next time step.
                h_filler.append(dh[0])
                c_filler.append(dh[1])

            # find the 5 best scores and indices amongst all the top 5 predictions for each word
            scores, indices = torch.topk(final_scores, k=beam_span)

            # the baseline scores for the next 5 predictions is set here
            final_scores = scores.repeat(1, beam_span)

            # storing the parent word to track back in the future
            parent_word.append(indices.detach().cpu().numpy() // beam_span)

            # map the hidden and cell states of the best predictions across each word in the batch, note that the hidden and cell states
            # corresponding to highest are not constant across a batch
            decoder_hidden_per_word = np.asarray([[h_filler[w.item() // beam_span][0][b] for w in indices[b]] for b in range(batch_size)])
            cell_per_word = np.asarray([[c_filler[w.item() // beam_span][0][b] for w in indices[b]] for b in range(batch_size)])
            eval_mode = True

            # store the actual indices of the 5 best words.
            indices = final_indices.gather(1, indices)
            # store the indices for future reference
            actual_indices.append(indices.detach().cpu().numpy())
            # Input to next time step are the 5 best words predicted by each word in the current batch so we will
            # have batch*beam size number of inputs
            decoder_inp = indices.clone()
            h_filler = []
            c_filler = []

        actual_indices = np.asarray(actual_indices)
        parent_word = np.asarray(parent_word)

        # find the index of the highest score
        highest_score_pos = torch.argmax(scores, 1)

        # iterate from last word to first word across all batches using the stored indices and parent nd array.
        prediction = []
        for i in range(len(actual_indices) - 1, -1, -1):
            # actual_indices[i] corresponds to the predictions in the ith time step, second index refers to each batch
            prediction.append(actual_indices[i, np.arange(batch_size), highest_score_pos])
            highest_score_pos = parent_word[i, np.arange(batch_size), highest_score_pos]

        # reverse the prediction to output from start to end
        prediction = prediction[::-1]
        print(prediction)
        for i in range(len(prediction)):
            print(dev_idx_to_word_q[str(prediction[i][0])])


def train(encoder: EncoderBILSTM, decoder: DecoderLSTM, epoch_count: int, train_loader: DataLoader, criterion, optimizer_enc: torch.optim.Optimizer, optimizer_dec: torch.optim.Optimizer, is_cuda: bool, teacher_forcing: bool = False, debug: bool = False, lr_schedule=False, start_epoch_at: int = 0):
    losses = []
    best_loss=1000000
    for epoch in range(start_epoch_at, start_epoch_at + epoch_count):
        total_batch_loss = 0
        for ind, batch in enumerate(train_loader):
            loss = 0
            questions, questions_org_len, answers, answers_org_len, pID = batch
            if questions.shape[1] > 1000:
                break
            if is_cuda:
                questions = questions.cuda()
                answers = answers.cuda()

            encoder_input, encoder_len = answers, answers_org_len
            decoder_input, decoder_len = questions, questions_org_len

            if is_cuda:
                encoder_out, encoder_hidden = encoder(encoder_input, torch.LongTensor(encoder_len).cuda(), False)
                encoder_len = torch.FloatTensor(encoder_len)
                if not teacher_forcing:
                    decoder_inp = torch.ones((len(questions), 1), dtype=torch.long).cuda()
            else:
                encoder_out, encoder_hidden = encoder(encoder_input, torch.LongTensor(encoder_len), False)
                encoder_len = torch.FloatTensor(encoder_len)
                if not teacher_forcing:
                    decoder_inp = torch.ones((len(questions), 1), dtype=torch.long)

            if teacher_forcing:
                decoder_out, decoder_hidden,attn_scores = decoder(decoder_input[:,:-1], encoder_hidden, encoder_out, encoder_len,False)
                decoder_out = decoder_out.transpose(0, 1).contiguous()
                decoder_out = decoder_out.transpose(1, 2).contiguous()
                loss = criterion(decoder_out, questions[:, :-1])
            else:
                decoder_hidden = (encoder_hidden[0].clone(), encoder_hidden[1].clone())
                eval_mode = False
                for j in range(questions.shape[1]):
                    decoder_out, decoder_hidden,attn_scores = decoder(decoder_inp, decoder_hidden, encoder_out, encoder_len, eval_mode=eval_mode)
                    # obtaining log_softmax scores we need to minimize log softmax over a span.
                    decoder_out = decoder_out.squeeze(0)
                    prediction = torch.argmax(decoder_out, 1).unsqueeze(1)
                    loss_val = criterion(decoder_out, questions[:, j])
                    loss += loss_val/questions.shape[1]
                    decoder_inp = prediction.clone().detach()
                    eval_mode = True

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()


            loss.backward()
            clip_grad_norm_(encoder.parameters(), 5)
            clip_grad_norm_(decoder.parameters(), 5)
            optimizer_enc.step()
            optimizer_dec.step()

            if lr_schedule:
                optimizer_enc = exp_lr_scheduler(optimizer_enc, epoch, lr_decay_epoch=8)
                optimizer_dec = exp_lr_scheduler(optimizer_dec, epoch, lr_decay_epoch=8)

            total_batch_loss += loss.item()
            if debug: print("Batch Loss: %f" % loss.item())
            if ind%1000==0:
                print("Batch %d Loss: %f" %(ind,loss.item()))
        losses.append(total_batch_loss)

        print("Epoch[%d] Loss: %f" % (epoch, total_batch_loss))
        if total_batch_loss<best_loss:
            torch.save(encoder.state_dict(), "model_weights/%d-encoder-SGD-small.pth" % epoch)
            torch.save(decoder.state_dict(), "model_weights/%d-decoder-SGD-small.pth" % epoch)
            best_loss=total_batch_loss
    torch.save(encoder.state_dict(), "model_weights/final-encoder-SGD-small.pth")
    torch.save(decoder.state_dict(), "model_weights/final-decoder-SGD-small.pth")
    return losses

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def show_attention(input_sentence, output_words, attentions):
    input_sentence = input_sentence[0]
    output_words=output_words[0]
    print(input_sentence)
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #plt.imshow()
    plt.show()
    plt.close()

def main(use_cuda=True):
    num_epoch = 15
    batch_size = 64
    use_cuda = use_cuda and torch.cuda.is_available()
    prev_model_weights_file_encoder = "model_weights/24-encoder-small_b128_mom.pth"
    prev_model_weights_file_decoder = "model_weights/24-decoder-small_b128_mom.pth"

    train_dataset = SquadDataset(split="train")

    train_vocab_size_sent = len(train_dataset.get_answer_word_to_idx())
    train_vocab_size_q = len(train_dataset.get_question_word_to_idx())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    word_embeddings_glove_q = GloVeEmbeddings.load_glove_embeddings(True)
    word_embeddings_glove_sent = GloVeEmbeddings.load_glove_embeddings(False)

    encoder = EncoderBILSTM(vocab_size=train_vocab_size_sent, embedding_dim=300, hidden_dim=600, n_layers=2, dropout=0.3, embeddings=word_embeddings_glove_sent)
    decoder = DecoderLSTM(vocab_size=train_vocab_size_q, embedding_dim=300, hidden_dim=600, n_layers=2, dropout=0.3, encoder_hidden_dim=600, embeddings=word_embeddings_glove_q)

    if prev_model_weights_file_encoder and prev_model_weights_file_decoder:
        encoder.load_state_dict(torch.load(prev_model_weights_file_encoder))
        decoder.load_state_dict(torch.load(prev_model_weights_file_decoder))

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer_enc = torch.optim.SGD(encoder.parameters(), lr=1.0)
    optimizer_dec = torch.optim.SGD(decoder.parameters(), lr=1.0)

    if not os.path.isdir("model_weights"):
        os.makedirs("model_weights", exist_ok=True)

    losses = train(encoder=encoder, decoder=decoder, epoch_count=num_epoch, train_loader=train_loader, criterion=criterion, optimizer_enc=optimizer_enc, optimizer_dec=optimizer_dec, is_cuda=use_cuda, debug=False, lr_schedule=True, start_epoch_at=0)
    plot_losses(losses)

    dev_dataset = SquadDataset(split="dev")
    print("-" * 100)
    given_sentence, ground_truth, prediction = greedy_search(encoder=encoder, decoder=decoder, dataset=train_dataset, use_cuda=use_cuda, batch_size=batch_size)
    print("-" * 100)
    print("Train BLEU CORPUS Score:: %f" % agg_bleu(ground_truth, prediction))


    print("Train BLEU1 Score:: %f" % BleuScorer.score_bleu1(ground_truth, prediction))
    print("Train BLEU2 Score:: %f" % BleuScorer.score_bleu2(ground_truth, prediction))
    print("Train BLEU3 Score:: %f" % BleuScorer.score_bleu3(ground_truth, prediction))
    print("Train BLEU4 Score:: %f" % BleuScorer.score_bleu4(ground_truth, prediction))

    print("=" * 100)
    given_sentence, ground_truth, prediction = greedy_search(encoder=encoder, decoder=decoder, dataset=dev_dataset, use_cuda=use_cuda, batch_size=batch_size)
    print("-" * 100)
    print("Dev BLEU CORPUS Score:: %f" % BleuScorer.corpus_score(ground_truth, prediction))
    print("Dev BLEU1 Score:: %f" % BleuScorer.score_bleu1(ground_truth, prediction))
    print("Dev BLEU2 Score:: %f" % BleuScorer.score_bleu2(ground_truth, prediction))
    print("Dev BLEU3 Score:: %f" % BleuScorer.score_bleu3(ground_truth, prediction))
    print("Dev BLEU4 Score:: %f" % BleuScorer.score_bleu4(ground_truth, prediction))
    print("-" * 100)


if __name__ == '__main__':
    main()
