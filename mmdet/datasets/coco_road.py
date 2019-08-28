from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoRoadDataset(CocoDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'train',
               'truck', 'traffic_light', 'stop_sign', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe')
