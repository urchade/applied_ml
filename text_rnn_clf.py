import torch
from torch import nn
from transformers import AutoModel

from blocks import RNNs, Classifier, Conv1Df


class TextRNN(nn.Module):
    def __init__(self, num_classes, embedding_dim, n_embeddings=None,
                 rnn_hidden_size=64, rnn_type='LSTM',
                 num_rnn_layers=1, bidirectional=False,
                 pooling='last_hidden', num_classifier_layers=2,
                 pt_weight=None, ffn_activation=nn.ReLU(),
                 pd_idx=0, rnn_dropout=0.0, clf_dropout=None):
        """
        Implementation of RNN-based text classifier
        Parameters
        ----------
        num_classes: int
            Number of classes
        embedding_dim: int
            Input dim for the RNN
        n_embeddings: int or None
            Number of word in the vocabulary.
            If None => no embedding layer added.
        num_classifier_layers: int
            Number of layer for the classifier
        rnn_hidden_size: int
            RNN hidden size
        rnn_type: str
            choice between 'LSTM', 'GRU' or 'RNN'. Default: 'LSTM'
        num_rnn_layers: int
            Number of layers for the RNN
        bidirectional: bool
            Bi-RNN if True.
        pt_weight: Tensor
            Weight for the embedding layer
        ffn_activation: nn.Module
            activation function for the FFN layers
        pooling: str
            pooling strategy for the RNN.
            Choice between 'last_hidden', 'max' or 'mean'
        pd_idx: int
            padding index of the embedding layer
        rnn_dropout: float
            Dropout rate for the RNN
        clf_dropout: float
            Dropout rate for the FFN layers.
        """

        super().__init__()

        self.rnn_type = rnn_type
        self.pooling = pooling
        self.len_vocab = n_embeddings

        direction = 2 if bidirectional else 1

        if self.len_vocab is not None:
            self.embedding = nn.Embedding(num_embeddings=n_embeddings,
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
                                     clf_dropout=clf_dropout, activation=ffn_activation)

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
    def __init__(self, n_out, len_vocab,
                 embedding_dim, channels=[16, 32, 64],
                 kernel_sizes=[3, 3, 3], last_pooling='mean',
                 pad_idx=0, weight=None,
                 activation=nn.ReLU(),
                 dropout=0.1):

        super().__init__()
        self.out_chan = channels[-1]
        self.last_pooling = last_pooling

        self.len_vocab = len_vocab

        if self.len_vocab is not None:
            self.embedding = nn.Embedding(len_vocab, embedding_dim,
                                          padding_idx=pad_idx, _weight=weight)

        self.conv = Conv1Df(in_channels=embedding_dim, channels=channels,
                            kernel_sizes=kernel_sizes, act=activation,
                            dropout=dropout)

        self.classifier = nn.Linear(channels[-1], n_out)

    def forward(self, x):
        if self.len_vocab:
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


class TextRNNConv(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=50, rnn_hidden=64,
                 rnn_type='LSTM', rnn_layer=1, cnn_pooling='max', bidirectional_rnn=False, conv_channels=[16, 32, 64],
                 conv_kernel_sizes=[3, 3, 3], pad_idx=None, emb_weight=None, rnn_dropout=0.0, clf_dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_idx, _weight=emb_weight)

        self.rnn = RNNs(rnn_type=rnn_type, input_size=embedding_dim, hidden_size=rnn_hidden, num_layers=rnn_layer,
                        bidirectional=bidirectional_rnn, dropout=rnn_dropout, batch_first=True)
        # Conv1Df(in_channels=embedding_dim, channels=conv_channels, kernel_sizes=conv_kernel_sizes)

        direction = 2 if bidirectional_rnn else 1

        self.cnn = TextCNN(num_classes, None, direction * rnn_hidden, conv_channels,
                           conv_kernel_sizes, cnn_pooling, dropout=clf_dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.rnn(x)[0]
        return self.cnn(x)


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
                                     clf_dropout=dropout, activation=activation)

    def forward(self, x):
        x = self.emb(x)  # (batch_size, seq_length, emb_dim)
        x = x.mean(dim=1)  # (batch_size, emb_dim)
        x = self.classifier(x)
        return x


class BertClassifier(nn.Module):
    def __init__(self, bert_name, num_classes, pooling='mean', clf_dropout=0.1, n_layers=1, act=nn.ReLU()):
        """
        Parameters
        Bert-base text classifier.
        ----------
        bert_name: str
            Name of the bert model. Ex = 'bert-base-uncased'
        num_classes: int
            Number of output classe
        pooling: str
            Pooling strategy for the classification.
            Choice in ['cls', 'max', 'mean']. Default: 'mean'
        clf_dropout: float
            Dropout rate in the FFN layers. Default: 0.1
        n_layers: int
            Number of FFN layers on top of bert. Default: 1
        act: nn.Module
            Activation function the FFNs. Default: nn.ReLU()
        """
        super().__init__()

        self.pooling = pooling
        self.bert = AutoModel.from_pretrained(bert_name)

        hidden_size = self.bert.config.hidden_size

        self.classifier = Classifier(hidden_size, num_classes, n_hidden_layers=n_layers,
                                     clf_dropout=clf_dropout, activation=act)

    def forward(self, x, att_mask=None):
        """
        Parameters
        ----------
        x: Input IDs
        att_mask: Attention mask
        Returns
        -------
        Logits
        """
        out, cls = self.bert(x, att_mask)

        if self.pooling == 'mean':
            h = out.mean(1)
        elif self.pooling == 'max':
            h = torch.max(out, dim=1)[0]
        elif self.pooling == 'cls':
            h = cls

        return self.classifier(h)
