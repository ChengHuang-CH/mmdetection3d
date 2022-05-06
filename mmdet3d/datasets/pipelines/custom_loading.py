from ..builder import PIPELINES
import open3d as o3d
import numpy as np
from mmdet3d.core.points import BasePoints, get_points_type
import json

@PIPELINES.register_module()
class LoadCustomImageFile(object):
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img_info']['img_filename']
        return results


@PIPELINES.register_module()
class LoadCustomLidarFile(object):
    def __init__(self,
                 coord_type):
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type

    def _load_points(self, pts_filename):
        cloud = o3d.io.read_point_cloud(pts_filename)
        point = np.asarray(cloud.points)  # nx3
        p_num = point.shape[0]
        point_append = np.zeros((20000 - p_num, 3))
        point = np.vstack((point, point_append))  # nx3 --> 20000x3

        return point

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        points = self._load_points(results['pts_filename'])
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[0])
        results['points'] = points
        return results


@PIPELINES.register_module()
class LoadCustomAnnotation(object):
    """Load Annotations
        """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_bbox_2d=False,
                 with_label_2d=False,
                 with_mask=False):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_bbox_2d = with_bbox_2d
        self.with_label_2d = with_label_2d

    @staticmethod
    def _decode_annotation_file(lidar_label_file, image_label_file):
        image_annotation = []
        lidar_annotation = []
        with open(lidar_label_file, 'r') as lidarLabel:
            lines = lidarLabel.readlines()
            for line in lines:
                line_list = json.loads(line)
                label_id = line_list[0]
                label_cls = line_list[1]
                bbx_vertices = line_list[2]
                lidar_annotation.append([label_cls, bbx_vertices])

        with open(image_label_file, 'r') as imgLabel:
            lines1 = imgLabel.readlines()
            for line in lines1:
                line_list1 = json.loads(line)

                idx, cls, min_x, min_y, max_x, max_y = line_list1
                image_annotation.append([cls, [min_x, min_y, max_x, max_y]])

        return lidar_annotation, image_annotation

    def _load_bboxes(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        annos = results['ann_info']

        results['gt_bboxes_3d'] = annos['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)

        return results
