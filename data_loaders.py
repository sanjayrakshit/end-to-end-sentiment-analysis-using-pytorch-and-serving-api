import pickle

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def load_pickle_datset(path):
    """
    Function to load a pickled data
    :param path: The path of the data
    :return: The loaded object
    """
    return pickle.load(open(path, 'rb'))


class MovieDataset(Dataset):
    def __init__(self, path):
        """
        This is the heart of the Dataset class.
        1. Loading the data
        2. Tokenizing the text
        3. Padding
        4. Converting to ids
        5. Converting to tensors
        :param path: Path of the dataset
        """
        self.tokenizer = BertTokenizer.from_pretrained('./bert-tokenizer')
        MAX_LEN = 512
        data = load_pickle_datset(path)
        self.x = data['review'].tolist()
        self.input_lengths = []
        for i, v in enumerate(self.x):
            truncated = self.tokenizer.tokenize(v)[:MAX_LEN - 2]
            self.input_lengths.append(len(truncated) + 2)
            v = ['[CLS]'] + truncated + ['[SEP]']
            v = v + ['[PAD]'] * (MAX_LEN - len(v))
            self.x[i] = self.tokenizer.convert_tokens_to_ids(v)

        self.x = torch.tensor(self.x)
        self.y = torch.tensor(data['sentiment'].values)
        self.input_lengths = torch.tensor(self.input_lengths)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx], self.input_lengths[idx]
