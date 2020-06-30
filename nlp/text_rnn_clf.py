import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_feature, num_classes, hidden_units,
                 n_hidden_layers, clf_dropout=None, activation=nn.ReLU()):
        super().__init__()

        classifier = []

        for i in range(n_hidden_layers - 1):
            classifier.append(nn.Linear(in_feature, hidden_units))
            classifier.append(activation)
            if clf_dropout:
                classifier.append(nn.Dropout(clf_dropout))
            in_feature = hidden_units

        classifier.append(nn.Linear(hidden_units, num_classes))

        self.fully_connected = nn.Sequential(*classifier)

    def forward(self, x):
        return self.fully_connected(x)


class RNNs(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, bidirectional, dropout, batch_first):
        super().__init__()

        configs = {'input_size': input_size, 'hidden_size': hidden_size, 'num_layers': num_layers,
                   'bidirectional': bidirectional, 'batch_first': batch_first, 'dropout': dropout}

        rnn_dict = {'LSTM': nn.LSTM(**configs),
                    'GRU': nn.GRU(**configs),
                    'RNN': nn.RNN(**configs)}

        self.rnn = rnn_dict[rnn_type]

    def forward(self, x):
        return self.rnn(x)


class Conv1Df(nn.Module):
    def __init__(self, in_channels, channels=[8, 16, 32], kernel_sizes=[2, 3, 4], act=nn.ReLU()):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        self.layers = nn.ModuleList()

        for out_channel, kernel in zip(channels, kernel_sizes):
            conv_layer = nn.Conv1d(in_channels, out_channel, kernel, padding=int((kernel - 1) / 2))
            norm = nn.BatchNorm1d(out_channel)
            layer = [conv_layer, act, norm]
            self.layers.append(nn.Sequential(*layer))
            in_channels = out_channel

    def forward(self, x):

        for i, layer in enumerate(self.layers):

            condition = i > 1 + i < len(self.layers)

            if condition:
                x = layer(x) + x  # Skip connection
            else:
                x = layer(x)

        return x


class TextRNN(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim,
                 num_classifier_layers=2, hidden_units=128,
                 rnn_hidden_size=64, rnn_type='LSTM',
                 num_rnn_layers=1, bidirectional=False,
                 pt_weight=None, activation=nn.ReLU(),
                 pooling='last_hidden', pd_idx=0,
                 rnn_dropout=0.0, clf_dropout=None):

        super().__init__()

        self.rnn_type = rnn_type
        self.pooling = pooling

        direction = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
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

        self.classifier = Classifier(in_feature=self.out_rnn, hidden_units=hidden_units,
                                     num_classes=num_classes, n_hidden_layers=num_classifier_layers,
                                     clf_dropout=clf_dropout, activation=activation)

    def forward(self, x):
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
                 kernel_sizes=[3, 3, 3], last_pooling='mean', pad_idx=0, weight=None, activation=nn.ReLU()):
        super().__init__()
        self.out_chan = channels[-1]
        self.last_pooling = last_pooling
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=weight)
        self.conv = Conv1Df(in_channels=embedding_dim, channels=channels, kernel_sizes=kernel_sizes, act=activation)
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
                 n_outputs, hidden_size=None,
                 activation=nn.ReLU(), dropout=0.0,
                 pad_idx=0, weight=None):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=emb_dim,
                                padding_idx=pad_idx, _weight=weight)

        in_features = emb_dim

        if hidden_size is None:
            hidden_size = emb_dim

        self.classifier = Classifier(in_features, n_outputs,
                                     hidden_size, n_hidden_layers=n_layers,
                                     clf_dropout=dropout, activation=activation)

    def forward(self, x):
        x = self.emb(x)  # (batch_size, seq_length, emb_dim)
        x = x.mean(dim=1)  # (batch_size, emb_dim)
        x = self.classifier(x)
        return x  # (batch_size, n_outputs)
