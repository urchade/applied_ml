import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TextPreprocessor:
    def __init__(self, train_path, sequence_length=70):

        # Storing some variables
        self.seq_len = sequence_length
        self.train_path = train_path
        self.pad_idx = 0  # putting padding index at 0

        print('Loading Spacy ....')
        # disable=['ner', 'tagger', 'parser'] for faster tokenization
        self.nlp = spacy.load('fr_core_news_sm', disable=['ner', 'tagger', 'parser'])

        self.train = pd.read_csv(train_path)

        self.n_classes = len(np.unique(self.train['Label'].values))

        self.label_encode = LabelEncoder()
        self.label_encode.fit(self.train['Label'].values)

        print(f'Number of classes: {self.n_classes}')

        self.id2word, self.word2id = self.get_vocab_dicts()

        self.num_words = len(self.id2word)

        print(f'Number of unique words: {len(self.id2word)}')

    def get_vocab_dicts(self):
        """
        Returns
        -------
        Return word dictionaries using the training set

        """
        text_list = self.train['Texte'].apply(lambda x: [str(a.lemma_).lower() for a in self.nlp(x)
                                                         if not (a.is_stop or not a.is_alpha)])

        id2word = dict(enumerate(set([a for s in text_list for a in s]), start=1))
        id2word[0] = '<padding>'
        word2id = {ids: word for word, ids in id2word.items()}
        return id2word, word2id

    def pad_sequence(self, list_ids):
        """
        Parameters
        ----------
        list_ids: A list of sequence of ids

        Returns
        -------
        Padded array
        """
        padded_array = np.ones(self.seq_len, dtype=np.int) * self.pad_idx
        for i, el in enumerate(list_ids[:self.seq_len]):
            padded_array[i] = el
        return padded_array

    def pad_mask(self, list_ids):
        """
        Parameters
        ----------
        list_ids: A list of padded sequence

        Returns
        -------
        Padding mask
        """
        return list(map(lambda a: 1 if a in [self.pad_idx] else 0, list_ids))

    def convert2idx(self, word_list):
        """
        Parameters
        ----------
        word_list: a list of tokens

        Returns
        -------
        A list of ids
        """
        lists = []
        for word in word_list:
            try:
                lists.append(self.word2id[word])
            except KeyError:
                # Just pass if the word is not in the dictionary (OOV)
                pass
        return lists

    def data2ids(self, data):
        """
        Apply tokenization, lemmatization, numericalization and padding.
        Parameters
        ----------
        data: pd.Series
        Returns
        -------
        text_ids, text_tokens
        """
        text_tokens = data.apply(lambda x: [str(a.lemma_).lower() for a in self.nlp(x)
                                            if not (a.is_stop or not a.is_alpha)])

        text_ids = text_tokens.apply(lambda x: self.convert2idx(x))
        text_ids = text_ids.apply(lambda x: self.pad_sequence(x))
        pad_mask = text_ids.apply(lambda x: self.pad_mask(x))
        return text_ids, text_tokens, pad_mask

    def get_bert_ids(self, text, tokenizer):
        text = text.apply(lambda x: x.lower())
        ids = text.apply(lambda x: tokenizer.encode(x, max_length=self.seq_len, pad_to_max_length=True))
        mask = ids.apply(lambda x: list(map(lambda a: 0 if a in [tokenizer.pad_token_id] else 1, x)))
        return ids, mask

    def get_data(self, path, tokenizer=None):
        """
        Returns
        -------
        The training data and its corresponding label
        """
        data = pd.read_csv(path)

        label = self.label_encode.transform(data['Label'])

        if tokenizer:
            ids, pad_mask = self.get_bert_ids(data['Texte'], tokenizer=tokenizer)
        else:
            ids, _, pad_mask = self.data2ids(data['Texte'])

        return ids, pad_mask, label


class NlpDataset(Dataset):
    def __init__(self, data: tuple):
        sequences, mask, labels = data

        self.x = torch.Tensor(sequences).long()
        self.y = torch.Tensor(labels).long()
        self.mask = torch.BoolTensor(mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.mask[item]
