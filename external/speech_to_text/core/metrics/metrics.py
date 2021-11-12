import editdistance


class CharacterErrorRate:
    def __init__(self):
        super().__init__()
        self.reset()

    def get_name(self) -> str:
        return "cer"

    def update(self, gt: str, pred: str) -> float:
        cur_score = editdistance.eval(gt, pred)
        cur_length = len(gt.label)
        self.score += cur_score
        self.length += cur_length
        return cur_score / cur_length

    def compute(self) -> float:
        return self.score / self.length if self.length != 0 else 0

    def reset(self) -> None:
        self.length, self.score = 0, 0


class WordErrorRate:
    def __init__(self):
        super().__init__()
        self.reset()

    def get_name(self) -> str:
        return "wer"

    def update(self, pred: str, gt: str) -> float:
        cur_score = editdistance.eval(gt.split(), pred.split())
        cur_words = len(gt.split())
        self.score += cur_score
        self.words += cur_words
        return cur_score / cur_words

    def compute(self) -> float:
        return self.score / self.words if self.words != 0 else 0

    def reset(self) -> None:
        self.words, self.score = 0, 0
