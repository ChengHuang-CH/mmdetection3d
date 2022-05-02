import tempfile
from os import path as osp

import mmcv
import numpy as np
import pyquaternion

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class CarlaUamDataset(Custom3DDataset):
    def __init__(self):
        super().__init__()
