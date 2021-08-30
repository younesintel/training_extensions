class BaseDecoder(object):
    def __call__(self, preds, preds_length):
        return self.decode(preds, preds_length)

    def decode(self, preds, preds_lengths):
        raise NotImplementedError
