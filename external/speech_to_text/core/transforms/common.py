class AudioCompose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data, sample_rate):
        for t in self.transforms:
            data, sample_rate = t(data, sample_rate)
        return data, sample_rate
