import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GlobalAttention(nn.Module):
    """
    Global Attention as described in 'Effective Approaches to Attention-based Neural Machine Translation'
    """

    def __init__(self, enc_hidden, dec_hidden):
        super(GlobalAttention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        # GENERAL ATTENTION: a = h_t^T W h_s (not concat)

        self.linear_in = nn.Linear(enc_hidden, dec_hidden, bias=False)

        # W [c, h_t]

        self.linear_out = nn.Linear(dec_hidden + enc_hidden, dec_hidden)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    @staticmethod
    def sequence_mask(lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        mask = (torch.arange(0, max_len).repeat(batch_size, 1)).lt(lengths.unsqueeze(1))
        if torch.cuda.is_available():
            return mask.cuda()
        return mask

    def forward(self, inputs, context, context_lengths):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        context_lengths (LongTensor): the source context lengths.
        """

        # (batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()

        mask = self.sequence_mask(context_lengths)

        # (batch, 1, src_len)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        # if next(self.parameters()).is_cuda:
        # mask = mask.cuda()
        align.data.masked_fill_(1 - mask, -float('inf'))  # fill <pad> with -inf

        align_vectors = self.softmax(align.view(batch * tgt_len, src_len))  # softmax over source scores
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden) -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(batch * tgt_len, self.enc_hidden + self.dec_hidden)
        attn_h = self.tanh(self.linear_out(concat_c).view(batch, tgt_len, self.dec_hidden))

        # transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        # (tgt_len, batch, dim)
        return attn_h, align_vectors

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()

        h_s = h_s.contiguous().view(src_batch * src_len, src_dim)
        h_s = self.linear_in(h_s)
        h_s = h_s.contiguous().view(src_batch, src_len, src_dim)
        # (batch, d, s_len)
        h_s = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s)


class EncoderBILSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float,
                 embeddings: np.array = None, n_layers: int = 1):
        super(EncoderBILSTM, self).__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm_dropout = dropout

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
            self.word_embeds.requires_grad=False

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim // 2, num_layers=self.n_layers,
                            bidirectional=True, dropout=self.lstm_dropout)

    def forward(self, inputs, lengths, return_packed=False):
        """
        Inputs:
            inputs: (seq_length, batch_size), non-packed inputs
            lengths: (batch_size)
        """
        # [seq_length, batch_size, embed_length]
        embeds = self.word_embeds(inputs)
        packed = pack_padded_sequence(embeds, lengths=lengths, batch_first=True)
        outputs, hiddens = self.lstm(packed)
        if not return_packed:
            return pad_packed_sequence(outputs, True)[0], hiddens
        return outputs, hiddens


class DecoderLSTM(nn.Module):
    """
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_layers: int = 1,
                 encoder_hidden_dim: int = None, embeddings: np.array = None,
                 dropout:float = 0.2):
        super(DecoderLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dropout=dropout
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        assert embeddings is not None
        self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embeds.requires_grad=False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,dropout=self.lstm_dropout)

        # h_t^T W h_s
        self.linear_out = nn.Linear(hidden_dim, vocab_size)
        self.attn = GlobalAttention(encoder_hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_lengths, eval_mode=False):
        """
        inputs: (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        embedded = self.word_embeds(inputs)
        embedded = embedded.transpose(0, 1)
        if not eval_mode:
            if self.n_layers==2:
                decode_hidden_init = torch.stack((torch.cat([hidden[0][0], hidden[0][1]],1),torch.cat([hidden[0][2], hidden[0][3]], 1)),0)
                decode_cell_init = torch.stack((torch.cat([hidden[1][0], hidden[1][1]],1),torch.cat([hidden[1][2], hidden[1][3]], 1)),0)
            else :
                decode_hidden_init = torch.cat([hidden[0][2], hidden[0][3]], 1).unsqueeze(0)
                decode_cell_init = torch.cat([hidden[1][2], hidden[1][3]], 1).unsqueeze(0)

        else:
            decode_hidden_init = hidden[0]
            decode_cell_init = hidden[1]


        # embedded = self.dropout(embedded)
        decoder_unpacked, decoder_hidden = self.lstm(embedded, (decode_hidden_init,decode_cell_init))
        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            decoder_unpacked.transpose(0, 1).contiguous(),  # (len, batch, d) -> (batch, len, d)
            context,  # (len, batch, d) -> (batch, len, d)
            context_lengths=context_lengths
        )

        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden