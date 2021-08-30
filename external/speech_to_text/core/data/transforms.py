import string
import torch
import torch.nn as nn


# Helpers

class ApplyByKey(object):
    def __init__(self, t, key):
        self.t = t
        self.key = key

    def __call__(self, sample):
        sample[self.key] = self.t(sample[self.key])
        return sample

class AddKey(object):
    def __init__(self, t, key, key_ext):
        self.t = t
        self.key = key
        self.key_ext = key_ext

    def __call__(self, sample):
        sample[self.key_ext] = self.t(sample[self.key])
        return sample

# Tensor Transforms

class TensorSqueeze(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, data):
        return data.squeeze(*self.args)

class TensorTranspose(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, data):
        return data.transpose(*self.args)

# Batch Transforms

class PadSequence(object):
    def __init__(self, padding_value=-1):
        self.padding_value = padding_value

    def __call__(self, data):
        data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=self.padding_value)
        return data

class ComputeLengths(object):
    def __call__(self, data):
        out = []
        for d in data:
            out.append(d.shape[0])
        out = torch.LongTensor(out)
        return out

# Text transforms

class TextPreprocess:
    def __init__(self, punctuation=string.punctuation + '—–«»−…‑'):
        self.punctuation = punctuation

    def __call__(self, data):
        data['text'] = data['text'].lower().strip().translate(str.maketrans('', '', self.punctuation))
        return data
