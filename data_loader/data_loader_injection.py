import pandas as pd
import torch
import pathlib
import numpy as np

from torch.utils.data import Dataset
from collections import Counter
from PIL import Image
from nltk.tokenize import word_tokenize


class dataLoader(Dataset):
    def __init__(self, root, transform):
        self.root = pathlib.Path('./datasets')
        self.word2idx = {}
        self.idx2word = {}
        self.transform = transform
        self.max_length_of_caption = 0

        path_txt = root / 'captions.txt'
        with open(path_txt) as f:
            data = pd.read_csv(f).to_numpy()

        with open('./attributes.txt') as f:
            attributes = pd.read_csv(f).to_numpy()

        attribute = np.delete(attributes, (np.where(attributes[:, 1] == ' ')), axis=0)
        data = np.delete(data, (np.where(attributes[:, 1] == ' ')), axis=0)

        self.file_name = data[:, 0]  # 1 img have 5 captions
        keywords = attribute[:, 1]
        captions = data[:, 1]

        counter = Counter()
        for caption in captions:
            tokens = word_tokenize(caption.lower())
            counter.update(tokens)

        self.words = []
        self.words.append('<pad>')
        self.words.append('<start>')
        self.words.append('<end>')
        self.words.append('<unk>')
        self.words.extend([word for word, cnt in counter.items() if cnt >= 1])

        for i, word in enumerate(self.words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        self.caps = []

        for cap in captions:
            tmp = []
            tokens = word_tokenize(cap.lower())
            if len(tokens) > self.max_length_of_caption:
                self.max_length_of_caption = len(tokens)
            tmp.extend([self.call_w2i(tok) for tok in tokens])
            self.caps.append(tmp)

        self.keywords = []
        for key in keywords:
            self.keywords.append(self.call_w2i(key))

    def __getitem__(self, i):

        path = self.root / 'Images' / f'{self.file_name[i]}'
        image = Image.open(path).convert('RGB')

        caption = []
        caption.append(self.word2idx['<start>'])
        caption.extend(self.caps[i])
        caption.append(self.word2idx['<end>'])

        caption = torch.LongTensor(caption)
        keyword = torch.LongTensor([self.keywords[i]])
        image = self.transform(image)

        return image, caption, keyword

    def __len__(self):
        return len(self.file_name)

    def call_w2i(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, keywords = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    keywords = torch.stack(keywords, 0)

    return images, targets, lengths, keywords




