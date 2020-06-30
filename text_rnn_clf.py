import torch
from torch import nn

from blocks import RNNs, Classifier, Conv1Df


class TextRNN(nn.Module):
    def __init__(self, num_classes, embedding_dim, len_vocab=None,
                 num_classifier_layers=2,
                 rnn_hidden_size=64, rnn_type='LSTM',
                 num_rnn_layers=1, bidirectional=False,
                 pt_weight=None, activation=nn.ReLU(),
                 pooling='last_hidden', pd_idx=0,
                 rnn_dropout=0.0, clf_dropout=None):

        super().__init__()

        self.rnn_type = rnn_type
        self.pooling = pooling
        self.len_vocab = len_vocab

        direction = 2 if bidirectional else 1

        if self.len_vocab is not None:
            self.embedding = nn.Embedding(num_embeddings=len_vocab,
                                          embedding_dim=embedding_dim,
                                          _weight=pt_weight,
                                          padding_idx=pd_idx)

        self.rnn_layer = RNNs(rnn_type=rnn_type, input_size=embedding_dim,
                              hidden_size=rnn_hidden_size, num_layers=num_rnn_layers,
                              bidirectional=bidirectional, dropout=rnn_dropout, batch_first=True)

        if self.pooling == 'last_hidden':
            self.out_rnn = num_rnn_layers * direction * rnn_hidden_size
        elif self.pooling == 'mean' or self.pooling == 'max':
            self.out_rnn = direction * rnn_hidden_size

        self.classifier = Classifier(in_feature=self.out_rnn,
                                     num_classes=num_classes, n_hidden_layers=num_classifier_layers,
                                     clf_dropout=clf_dropout, activation=activation)

    def forward(self, x):

        if self.len_vocab is not None:
            x = self.embedding(x)

        if self.rnn_type == 'LSTM':
            out, (h, _) = self.rnn_layer(x)
        else:
            out, h = self.rnn_layer(x)

        if self.pooling == 'last_hidden':
            y = h.view(-1, self.out_rnn)
        elif self.pooling == 'mean':
            y = torch.mean(out, dim=1)
        elif self.pooling == 'max':
            y = torch.max(out, dim=1)[0]

        return self.classifier(y)


class TextCNN(nn.Module):
    def __init__(self, n_out, num_embeddings, embedding_dim, channels=[16, 32, 64],
                 kernel_sizes=[3, 3, 3], last_pooling='mean', pad_idx=0, weight=None, activation=nn.ReLU(),
                 dropout=0.1):
        super().__init__()
        self.out_chan = channels[-1]
        self.last_pooling = last_pooling
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=weight)

        self.conv = Conv1Df(in_channels=embedding_dim, channels=channels,
                            kernel_sizes=kernel_sizes, act=activation,
                            dropout=dropout)

        self.classifier = nn.Linear(channels[-1], n_out)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.last_pooling == 'mean':
            x = x.mean(-1)
        elif self.last_pooling == 'max':
            x = torch.max(x, dim=-1)[0]
        return self.classifier(x)


class TextConvRNN(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=50, rnn_hidden=64,
                 rnn_type='LSTM', rnn_layer=1, rnn_pooling='max', bidirectional_rnn=False, conv_channels=[16, 32, 64],
                 conv_kernel_sizes=[3, 3, 3], pad_idx=None, emb_weight=None, rnn_dropout=0.0, clf_dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=emb_weight)

        self.conv = Conv1Df(in_channels=embedding_dim, channels=conv_channels, kernel_sizes=conv_kernel_sizes)

        self.rnn = TextRNN(num_classes, conv_channels[-1], num_classifier_layers=1, rnn_hidden_size=rnn_hidden,
                           rnn_type=rnn_type, num_rnn_layers=rnn_layer, bidirectional=bidirectional_rnn,
                           rnn_dropout=rnn_dropout, clf_dropout=clf_dropout, pooling=rnn_pooling)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.rnn(x)


class TextRNNCONV(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=50, rnn_hidden=64,
                 rnn_type='LSTM', rnn_layer=1, rnn_pooling='max', bidirectional_rnn=False, conv_channels=[16, 32, 64],
                 conv_kernel_sizes=[3, 3, 3], pad_idx=None, emb_weight=None, rnn_dropout=0.0, clf_dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=emb_weight)

        self.conv = Conv1Df(in_channels=embedding_dim, channels=conv_channels, kernel_sizes=conv_kernel_sizes)

        self.rnn = TextRNN(num_classes, conv_channels[-1], num_classifier_layers=1, rnn_hidden_size=rnn_hidden,
                           rnn_type=rnn_type, num_rnn_layers=rnn_layer, bidirectional=bidirectional_rnn,
                           rnn_dropout=rnn_dropout, clf_dropout=clf_dropout, pooling=rnn_pooling)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.rnn(x)


class AttentionBiLSTM(nn.Module):
    def __init__(self, n_classes, num_embeddings,
                 embedding_dim, hidden_size, weight=None,
                 rnn_dropout=0.1, att_dropout=0.3,
                 cls_dropout=0.3, rnn_type='LSTM',
                 bidirectional=True, num_heads=1, pad_idx=0):
        super().__init__()

        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=weight)

        self.rnn = RNNs(rnn_type=self.rnn_type, input_size=embedding_dim, hidden_size=hidden_size,
                        num_layers=1, bidirectional=bidirectional,
                        dropout=rnn_dropout, batch_first=True)

        self.biLSTM_hidden = 2 * hidden_size

        self.query = nn.Parameter(torch.randn(size=(self.biLSTM_hidden,)))

        self.mha = nn.MultiheadAttention(embed_dim=self.biLSTM_hidden, num_heads=num_heads, dropout=att_dropout)

        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(cls_dropout),
                                        nn.Linear(self.biLSTM_hidden, n_classes))

    def forward(self, x, mask=None):
        # dim of x: (batch_size, seq_len)
        out_emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # dim of out:  (batch_size, seq_len, 2 * hidden_size)
        out, *_ = self.rnn(out_emb)

        batch_size, seq_len, emb_dim = out.size()

        Q = self.query.expand(size=(1, batch_size, emb_dim))  # Same query for all batches
        V = out.transpose(0, 1)  # (seq_len, batch_size, 2 * hidden_size)
        K = out.transpose(0, 1)  # (seq_len, batch_size, 2 * hidden_size)

        att_out, _ = self.mha(Q, K, V, key_padding_mask=mask)

        # (1, batch_size, 2 * hidden_size)
        att_out = att_out.squeeze()  # (batch_size, 2 * hidden_size)

        out_linear = self.classifier(att_out)  # (batch_size, 2 * n_classes)

        return out_linear


class DAN(nn.Module):
    def __init__(self, n_emb, emb_dim, n_layers,
                 n_outputs,
                 activation=nn.ReLU(), dropout=0.0,
                 pad_idx=0, weight=None):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=emb_dim,
                                padding_idx=pad_idx, _weight=weight)

        in_features = emb_dim

        self.classifier = Classifier(in_features, n_outputs,
                                     n_hidden_layers=n_layers,
                                     clf_dropout=dropout, activation=activation, )

    def forward(self, x):
        x = self.emb(x)  # (batch_size, seq_length, emb_dim)
        x = x.mean(dim=1)  # (batch_size, emb_dim)
        x = self.classifier(x)
        return x  # (batch_size, n_outputs)
