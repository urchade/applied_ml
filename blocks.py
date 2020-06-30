from torch import nn


class Classifier(nn.Module):
    def __init__(self, in_feature, num_classes,
                 n_hidden_layers=1, clf_dropout=None, activation=nn.ReLU()):
        super().__init__()

        classifier = []

        for i in range(n_hidden_layers - 1):
            classifier.append(nn.Linear(in_feature, in_feature))
            classifier.append(activation)
            if clf_dropout:
                classifier.append(nn.Dropout(clf_dropout))

        classifier.append(nn.Linear(in_feature, num_classes))

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
    def __init__(self, in_channels, channels=[8, 16, 32], kernel_sizes=[2, 3, 4], act=nn.ReLU(), dropout=0.):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        self.layers = nn.ModuleList()

        for out_channel, kernel in zip(channels, kernel_sizes):
            conv_layer = nn.Conv1d(in_channels, out_channel, kernel, padding=int((kernel - 1) / 2))
            norm = nn.BatchNorm1d(out_channel)
            layer = [conv_layer, act, nn.Dropout(dropout), norm]
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
