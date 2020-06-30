from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data_prep import NlpDataset, TextPreprocessor
from text_rnn_clf import DAN
from trainers import ClassifierTrainer

prep = TextPreprocessor(train_path=r'data\data.train.csv', sequence_length=30)

feat = AutoModel.from_pretrained('camembert-base')
tok = AutoTokenizer.from_pretrained('camembert-base')

train_ = NlpDataset(prep.get_data(r'data\data.train.csv', tok))
valid_ = NlpDataset(prep.get_data(r'data\data.val.csv', tok))

train_loader = DataLoader(dataset=train_, batch_size=32)
valid_loader = DataLoader(dataset=valid_, batch_size=64)

model = DAN(n_emb=tok.vocab_size, emb_dim=feat.config.hidden_size,
            n_layers=1, n_outputs=19, dropout=0.1)

model.emb = feat.embeddings.word_embeddings

model.cuda()

trainer = ClassifierTrainer(model, train_loader, valid_loader, nn.CrossEntropyLoss())

trainer.train(max_epoch=20, patience=2)

trainer.plot_metric_curve()
