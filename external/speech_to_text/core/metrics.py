import editdistance


class WordErrorRate:
    def __init__(self):
        super().__init__()
        self.reset()

    def get_name(self):
        return "wer"

    def update(self, pred, gt):
        cur_score = editdistance.eval(gt.split(), pred.split())
        cur_words = len(gt.split())
        self.score += cur_score
        self.words += cur_words
        return cur_score / cur_words

    def compute(self):
        return self.score / self.words if self.words != 0 else 0

    def reset(self):
        self.words, self.score = 0, 0
