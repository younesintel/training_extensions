from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset


class AnomalyDataset(DatasetEntity):
    def __init__(
        self,
        train_ann_file=None,
        train_data_root=None,
        val_ann_file=None,
        val_data_root=None,
        test_ann_file=None,
        test_data_root=None,
    ):

        labels_list = []
        items = []
