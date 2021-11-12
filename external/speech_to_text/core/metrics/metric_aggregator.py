from typing import Dict


class MetricAggregator:
    def __init__(self, metrics=[]):
        self.metrics = metrics

    def update(self, pred: str, gt: str) -> Dict[str, float]:
        out = {}
        for m in self.metrics:
            out[m.get_name()] = m.update(pred, gt)
        return out

    def compute(self) -> Dict[str, float]:
        out = {}
        for m in self.metrics:
            out[m.get_name()] = m.compute(pred, gt)
        return out

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()
