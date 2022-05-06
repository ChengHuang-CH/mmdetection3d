import mmcv
import numpy as np
import glob
import os
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
import json


@DATASETS.register_module()
class CarlaUamDataset(Custom3DDataset):
    def _build_default_pipeline(self):
        pass

    CLASSES = ('Vehicle', 'Walker', 'Drone')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False
                 ):
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         pipeline=pipeline,
                         classes=classes,
                         modality=modality,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         test_mode=test_mode)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True
            )

            self.dataset_root = data_root

            self.subfolders = [name for name in os.listdir(self.dataset_root) if
                               os.path.isdir(os.path.join(self.dataset_root, name))]  # only keep folders

            self.subfolder_name = None

            # for traversing files in one folder
            self.img_folder = None
            self.img_files = None
            self.img = None

            self.lidar_folder = None
            self.lidar_files = None
            self.cloud = None

            # for files in all folders
            self.img_paths = []
            self.lidar_paths = []
            self.img_label_paths = []
            self.lidar_label_paths = []
            self.folder_ind = []

            self.lidar_label_folder = None
            self.lidar_label_files = None

            self.image_label_folder = None
            self.image_label_files = None

            self.config_params_file = None
            self.config_params = None

            self.lidar2world = None
            self.world2camera = None
            self.K = None

    def _load_one_folder(self, folder_i):

        self.subfolder_name = self.subfolders[folder_i]

        self.img_folder = f'{self.dataset_root}/{self.subfolder_name}/image'
        self.img_files = sorted(glob.glob(f'{self.img_folder}/*.png'))

        self.image_label_folder = f'{self.dataset_root}/{self.subfolder_name}/image_labels'
        self.image_label_files = sorted(glob.glob(f'{self.image_label_folder}/*.txt'))

        self.lidar_folder = f'{self.dataset_root}/{self.subfolder_name}/lidar'
        self.lidar_files = sorted(glob.glob(f'{self.lidar_folder}/*.ply'))

        self.lidar_label_folder = f'{self.dataset_root}/{self.subfolder_name}/lidar_labels'
        self.lidar_label_files = sorted(glob.glob(f'{self.lidar_label_folder}/*.txt'))

        self.config_params_file = f'{self.dataset_root}/{self.subfolder_name}/params.json'
        self.config_params = self._load_config_params()

        self.lidar2world = np.asarray(self.config_params['lidar2world'])
        self.world2camera = np.asarray(self.config_params['world2camera'])
        self.K = np.asarray(self.config_params['camera_intrinsic'])

        one_folder_dict = []
        for img_file, lidar_file, image_label_file, lidar_label_file in zip(self.img_files,
                                                                            self.lidar_files,
                                                                            self.image_label_files,
                                                                            self.lidar_label_files):
            one_dict = dict(
                lidar_filename=lidar_file,
                lidar_label_filename=lidar_label_file,
                img_filename=img_file,
                img_label_filename=image_label_file,
                lidar2world=self.lidar2world,
                world2camera=self.world2camera,
                intrinsic=self.K

            )

            # if self.modality['use_camera']:
            #     one_dict.update(dict())

            one_folder_dict.append(one_dict)

        return one_folder_dict

    def _load_config_params(self):
        with open(self.config_params_file, 'rb') as jsonFile:
            data = json.load(jsonFile)
        return data

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        data_infos = []
        for folder_i, subfolder in enumerate(self.subfolders):
            one_folder_dict = self._load_one_folder(folder_i=folder_i)

            data_infos += one_folder_dict

        # data_info: list(dict, dict, ...)
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        lidar2img = info['intrinsic'] @ info['world2camera'] @ info['lidar2world']

        input_dict = dict(
            pts_filename=info['lidar_filename'],
            img_info=dict(filename=info['img_filename']),
            lidar2img=lidar2img)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        anns_results = dict(
            gt_bboxes_3d=info['lidar_label_filename'],
            gt_bboxes_2d=info['img_label_filename']
        )

        return anns_results


if __name__ == '__main__':
    dataset = CarlaUamDataset(data_root='../../../../1_Datasets/carla_uam_dataset',
                              ann_file='../../../../1_Datasets/carla_uam_dataset')
