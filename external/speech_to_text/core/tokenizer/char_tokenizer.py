import torch
from tqdm.autonotebook import tqdm


class CharLevelTokenizer(object):
    def __init__(self, vocab=""):
        super().__init__()
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def __call__(self, s, to_tensor=True):
        return self.encode(s, to_tensor)

    def encode(self, s, to_tensor=True):
        out = []
        for c in s:
            out.append(self.vocab.find(c))
        if to_tensor:
            out = torch.LongTensor(out)
        return out

    def decode(self, tokens):
        return "".join([self.vocab[i] for i in tokens])

    def state_dict(self):
        return {"vocab": self.vocab}

    def load_state_dict(self, state_dict):
        self.vocab = state_dict["vocab"]

    def get_blank_symbol(self):
        return "#", self.encode("#", False)[0]

    def id_to_subword(self, idx):
        return self.vocab[idx]

    @staticmethod
    def train(dataset):
        vocab = set()
        for sample in tqdm(dataset):
            for i in list(sample["text"]):
                vocab.add(i)
        vocab = list(vocab)
        vocab.sort()
        vocab = ''.join(vocab) + "#"
        return vocab
