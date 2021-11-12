from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from anomalib.datasets.anomaly_dataset import make_dataset

class JSONFromDataset:
    def __init__(self, root: Union[str, Path]) -> None:
        self.root = root if isinstance(root, Path) else Path(root)

    def __enter__(self):
        data_frame = make_dataset(path=self.root)
        data_frame = data_frame[["image_path"]]
        


class AnomalyDataset(DatasetEntity):
    def __init__(
        self,
        train_ann_file: Union[str, Path],
        train_data_root: Union[str, Path],
        val_ann_file: Union[str, Path],
        val_data_root: Union[str, Path],
        test_ann_file: Optional[Union[str, Path]],
        test_data_root: Optional[Union[str, Path]],
    ):

        items = []

        self.normal_label = LabelEntity(name="normal", domain=Domain.ANOMALY_CLASSIFICATION)
        self.abnormal_label = LabelEntity(name="anomalous", domain=Domain.ANOMALY_CLASSIFICATION)

        train_ann_file = train_ann_file if isinstance(train_ann_file, Path) else Path(train_ann_file)
        train_data_root = train_data_root if isinstance(train_data_root, Path) else Path(train_data_root)
        val_ann_file = val_ann_file if isinstance(val_ann_file, Path) else Path(val_ann_file)
        val_data_root = val_data_root if isinstance(val_data_root, Path) else Path(val_data_root)

        if train_ann_file is not None or train_data_root is not None:
            items.extend(
                self.get_dataset_items(
                    ann_file_path=train_ann_file, data_root_dir=train_data_root, subset=Subset.TRAINING
                )
            )

        if val_ann_file is not None or val_data_root is not None:
            items.extend(
                self.get_dataset_items(
                    ann_file_path=val_ann_file, data_root_dir=val_data_root, subset=Subset.VALIDATION
                )
            )

        if test_ann_file is not None and test_data_root is not None:
            test_ann_file = test_ann_file if isinstance(test_ann_file, Path) else Path(test_ann_file)
            test_data_root = test_data_root if isinstance(test_data_root, Path) else Path(test_data_root)
            items.extend(
                self.get_dataset_items(ann_file_path=test_ann_file, data_root_dir=test_data_root, subset=Subset.TESTING)
            )

        super().__init__(items=items)

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path, subset: Subset) -> List[DatasetItemEntity]:
        test_mode = subset in {Subset.VALIDATION, Subset.TESTING}
        label: LabelEntity = self.abnormal_label
        if test_mode:
            label = self.normal_label
        # read annotation file
        samples = pd.read_json(path=ann_file_path)

        dataset_items = []
        for _, sample in samples.iterrows():
            # Create image
            image = Image(file_path=data_root_dir / sample.image_path)
            # Create annotation
            shape = Rectangle(x1=0, y1=0, x2=1, y2=1)
            labels = [ScoredLabel(label)]
            annotations = [Annotation(shape=shape, labels=labels)]
            annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)

            # Create dataset item
            dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=sample.subset)
            # Add to dataset items
            dataset_items.append(dataset_item)

        return dataset_items
