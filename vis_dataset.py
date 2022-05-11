# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction, mkdir_or_exist

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer.image_vis import (draw_camera_bbox3d_on_img,
                                               draw_depth_bbox3d_on_img,
                                               draw_lidar_bbox3d_on_img)
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from mmdet3d.datasets import build_dataset
import threading
import open3d as o3d
import open3d.visualization.gui as gui
import time


class VisDataset:
    def __init__(self, vis=True):
        """

        """
        # for open3d visualization
        self.vis = vis

        if self.vis:
            self.lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6],
                          [3, 7], ]

            self.bgr = [[142, 0, 0], [60, 20, 220], [0, 255, 0]]  # [vehicle, pedestrian, flights]
            self.rgb = [[0, 0, 142], [220, 20, 60], [0, 255, 0]]  # [vehicle, pedestrian, flights]

            self.app = None
            self.window = None
            self.scene1 = None
            self.scene2 = None
            self.scene3 = None
            self.progress = None
            self.slider = None
            self.progress_text = None
            self.intedit_h = None
            self.image_widget = None
            self.is_done = False
            self.bbx_num = 0
            self.play = False

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Browse a dataset')
        parser.add_argument('config', help='train config file path')
        parser.add_argument(
            '--skip-type',
            type=str,
            nargs='+',
            default=['Normalize'],
            help='skip some useless pipeline')
        parser.add_argument(
            '--output-dir',
            default=None,
            type=str,
            help='If there is no display interface, you can save it')
        parser.add_argument(
            '--task',
            type=str,
            choices=['det', 'seg', 'multi_modality-det', 'mono-det'],
            help='Determine the visualization method depending on the task.')
        parser.add_argument(
            '--aug',
            action='store_true',
            help='Whether to visualize augmented datasets or original dataset.')
        parser.add_argument(
            '--online',
            action='store_true',
            help='Whether to perform online visualization. Note that you often '
                 'need a monitor to do so.')
        parser.add_argument(
            '--cfg-options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
                 'in xxx=yyy format will be merged into config file. If the value to '
                 'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                 'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                 'Note that the quotation marks are necessary and that no white space '
                 'is allowed.')
        args = parser.parse_args()
        return args

    def build_data_cfg(self, config_path, skip_type, aug, cfg_options):
        """Build data config for loading visualization data."""

        cfg = Config.fromfile(config_path)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
        # extract inner dataset of `RepeatDataset` as `cfg.data.train`
        # so we don't need to worry about it later
        if cfg.data.train['type'] == 'RepeatDataset':
            cfg.data.train = cfg.data.train.dataset
        # use only first dataset for `ConcatDataset`
        if cfg.data.train['type'] == 'ConcatDataset':
            cfg.data.train = cfg.data.train.datasets[0]
        train_data_cfg = cfg.data.train

        if aug:
            show_pipeline = cfg.train_pipeline
        else:
            show_pipeline = cfg.eval_pipeline
            for i in range(len(cfg.train_pipeline)):
                if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                    show_pipeline.insert(i, cfg.train_pipeline[i])
                # Collect points as well as labels
                if cfg.train_pipeline[i]['type'] == 'Collect3D':
                    if show_pipeline[-1]['type'] == 'Collect3D':
                        show_pipeline[-1] = cfg.train_pipeline[i]
                    else:
                        show_pipeline.append(cfg.train_pipeline[i])

        train_data_cfg['pipeline'] = [
            x for x in show_pipeline if x['type'] not in skip_type
        ]

        return cfg

    def to_depth_mode(self, points, bboxes):
        """Convert points and bboxes to Depth Coord and Depth Box mode."""
        if points is not None:
            points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
        if bboxes is not None:
            bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR,
                                       Box3DMode.DEPTH)
        return points, bboxes

    def show_det_data(self, input, out_dir, show=False):
        """Visualize 3D point cloud and 3D bboxes."""
        img_metas = input['img_metas']._data
        points = input['points']._data.numpy()
        gt_bboxes = input['gt_bboxes_3d']._data.tensor
        if img_metas['box_mode_3d'] != Box3DMode.DEPTH:
            points, gt_bboxes = self.to_depth_mode(points, gt_bboxes)
        filename = osp.splitext(osp.basename(img_metas['pts_filename']))[0]
        self.show_result(
            points,
            gt_bboxes.clone(),
            None,
            out_dir,
            filename,
            show=show,
            snapshot=True)

    def show_proj_bbox_img(self, input, out_dir, show=False, is_nus_mono=False):
        """Visualize 3D bboxes on 2D image by projection."""
        gt_bboxes = input['gt_bboxes_3d']._data
        img_metas = input['img_metas']._data
        img = input['img']._data.numpy()
        # need to transpose channel to first dim
        img = img.transpose(1, 2, 0)
        # no 3D gt bboxes, just show img
        if gt_bboxes.tensor.shape[0] == 0:
            gt_bboxes = None
        filename = Path(img_metas['filename']).name
        if isinstance(gt_bboxes, DepthInstance3DBoxes):
            self.show_multi_modality_result(
                img,
                gt_bboxes,
                None,
                None,
                out_dir,
                filename,
                box_mode='depth',
                img_metas=img_metas,
                show=show)
        elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
            self.show_multi_modality_result(
                img,
                gt_bboxes,
                None,
                img_metas['lidar2img'],
                out_dir,
                filename,
                box_mode='lidar',
                img_metas=img_metas,
                show=show)
        elif isinstance(gt_bboxes, CameraInstance3DBoxes):
            self.show_multi_modality_result(
                img,
                gt_bboxes,
                None,
                img_metas['cam2img'],
                out_dir,
                filename,
                box_mode='camera',
                img_metas=img_metas,
                show=show)
        else:
            # can't project, just show img
            warnings.warn(
                f'unrecognized gt box type {type(gt_bboxes)}, only show image')
            self.show_multi_modality_result(
                img, None, None, None, out_dir, filename, show=show)

    def show_multi_modality_result(self,
                                   img,
                                   gt_bboxes,
                                   pred_bboxes,
                                   proj_mat,
                                   out_dir,
                                   filename,
                                   box_mode='lidar',
                                   img_metas=None,
                                   show=False,
                                   gt_bbox_color=(61, 102, 255),
                                   pred_bbox_color=(241, 101, 72)):
        """Convert multi-modality detection results into 2D results.

        Project the predicted 3D bbox to 2D image plane and visualize them.

        Args:
            img (np.ndarray): The numpy array of image in cv2 fashion.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
            pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
            proj_mat (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            out_dir (str): Path of output directory.
            filename (str): Filename of the current frame.
            box_mode (str, optional): Coordinate system the boxes are in.
                Should be one of 'depth', 'lidar' and 'camera'.
                Defaults to 'lidar'.
            img_metas (dict, optional): Used in projecting depth bbox.
                Defaults to None.
            show (bool, optional): Visualize the results online. Defaults to False.
            gt_bbox_color (str or tuple(int), optional): Color of bbox lines.
               The tuple of color should be in BGR order. Default: (255, 102, 61).
            pred_bbox_color (str or tuple(int), optional): Color of bbox lines.
               The tuple of color should be in BGR order. Default: (72, 101, 241).
        """
        if box_mode == 'depth':
            draw_bbox = draw_depth_bbox3d_on_img
        elif box_mode == 'lidar':
            draw_bbox = draw_lidar_bbox3d_on_img
        elif box_mode == 'camera':
            draw_bbox = draw_camera_bbox3d_on_img
        else:
            raise NotImplementedError(f'unsupported box mode {box_mode}')

        result_path = osp.join(out_dir, filename)
        mmcv.mkdir_or_exist(result_path)

        if show:
            show_img = img.copy()
            if gt_bboxes is not None:
                show_img = draw_bbox(
                    gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
            if pred_bboxes is not None:
                show_img = draw_bbox(
                    pred_bboxes,
                    show_img,
                    proj_mat,
                    img_metas,
                    color=pred_bbox_color)
            mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

        if img is not None:
            mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

        if gt_bboxes is not None:
            gt_img = draw_bbox(
                gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
            mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

        if pred_bboxes is not None:
            pred_img = draw_bbox(
                pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
            mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))

    def show_result(self,
                    points,
                    gt_bboxes,
                    pred_bboxes,
                    out_dir,
                    filename,
                    show=False,
                    snapshot=False,
                    pred_labels=None):
        """Convert results into format that is directly readable for meshlab.

        Args:
            points (np.ndarray): Points.
            gt_bboxes (np.ndarray): Ground truth boxes.
            pred_bboxes (np.ndarray): Predicted boxes.
            out_dir (str): Path of output directory
            filename (str): Filename of the current frame.
            show (bool, optional): Visualize the results online. Defaults to False.
            snapshot (bool, optional): Whether to save the online results.
                Defaults to False.
            pred_labels (np.ndarray, optional): Predicted labels of boxes.
                Defaults to None.
        """
        result_path = osp.join(out_dir, filename)
        mmcv.mkdir_or_exist(result_path)

        if show:


            vis = Visualizer(points)
            if pred_bboxes is not None:
                if pred_labels is None:
                    vis.add_bboxes(bbox3d=pred_bboxes)
                else:
                    palette = np.random.randint(
                        0, 255, size=(pred_labels.max() + 1, 3)) / 256
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(pred_bboxes[j])
                    for i in labelDict:
                        vis.add_bboxes(
                            bbox3d=np.array(labelDict[i]),
                            bbox_color=palette[i],
                            points_in_box_color=palette[i])

            if gt_bboxes is not None:
                vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
            show_path = osp.join(result_path,
                                 f'{filename}_online.png') if snapshot else None
            vis.show(show_path)

    def main(self):
        args = self.parse_args()

        if args.output_dir is not None:
            mkdir_or_exist(args.output_dir)

        cfg = self.build_data_cfg(args.config, args.skip_type, args.aug,
                                  args.cfg_options)
        try:
            dataset = build_dataset(
                cfg.data.train, default_args=dict(filter_empty_gt=False))
        except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
            dataset = build_dataset(cfg.data.train)

        dataset_type = cfg.dataset_type
        # configure visualization mode
        vis_task = args.task  # 'det', 'seg', 'multi_modality-det', 'mono-det'
        progress_bar = mmcv.ProgressBar(len(dataset))

        for input in dataset:
            if vis_task in ['det', 'multi_modality-det']:
                # show 3D bboxes on 3D point clouds
                self.show_det_data(input, args.output_dir, show=args.online)
            if vis_task in ['multi_modality-det', 'mono-det']:
                # project 3D bboxes to 2D image
                self.show_proj_bbox_img(
                    input,
                    args.output_dir,
                    show=args.online,
                    is_nus_mono=(dataset_type == 'NuScenesMonoDataset'))

            progress_bar.update()

    """
        below functions for open3d visualization
        """

    def _on_layout(self, theme):
        r = self.window.content_rect
        self.scene1.frame = gui.Rect(r.x, r.y, r.width / 2, r.height / 5 * 4)
        self.scene2.frame = gui.Rect(r.x + r.width / 2 + 1, r.y, r.width / 2, r.height / 5 * 4)
        self.scene3.frame = gui.Rect(r.x, r.y + r.height / 5 * 4 + 1, r.width, r.height / 5)

    def _slider_value_changed(self, value):
        self.progress.value = (self.slider.int_value + 1) / len(self.img_files)
        self.progress_text.text = f'({self.slider.int_value}/{len(self.img_files) - 1})'
        self.intedit_h.int_value = self.slider.int_value

    def _on_main_window_closing(self):
        self.is_done = True
        return True

    def _play_btn_clicked(self):
        self.play = True

    def _pause_btn_clicked(self):
        self.play = False

    def _intedit_value_changed(self, value):
        self.slider.int_value = self.intedit_h.int_value
        self.progress.value = (self.slider.int_value + 1) / len(self.img_files)
        self.progress_text.text = f'({self.slider.int_value}/{len(self.img_files) - 1})'
        self.intedit_h.int_value = self.slider.int_value

    def _get_lidar_img(self, frame_i):
        lidar_label_file = self.lidar_label_files[frame_i]
        image_label_file = self.image_label_files[frame_i]
        lidar_annotation, image_annotation = self._decode_annotation_file(lidar_label_file=lidar_label_file,
                                                                          image_label_file=image_label_file)
        self.img = cv2.imread(self.img_files[frame_i])  # opencv read image as BGR mode
        self.cloud = o3d.io.read_point_cloud(self.lidar_files[frame_i])

        lidar_box_set = []
        for l_labels in lidar_annotation:
            l_cls, l_bbx = l_labels
            # for open3d bounding box visualization
            color = [[ci / 255.0 for ci in self.rgb[l_cls]] for _ in range(len(self.lines))]
            o3d_vertices = o3d.utility.Vector3dVector(l_bbx)
            line_set = o3d.geometry.LineSet(
                points=o3d_vertices,
                lines=o3d.utility.Vector2iVector(self.lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(color)
            lidar_box_set.append(line_set)
        # o3d.visualization.draw_geometries([self.cloud] + lidar_box_set)

        for i_labels in image_annotation:
            i_cls, [min_x, min_y, max_x, max_y] = i_labels
            self.img = cv2.rectangle(self.img, (min_x, min_y), (max_x, max_y), self.bgr[i_cls], 2)

        return lidar_box_set

    def _update_next_frame(self, frame_i):
        try:
            self.scene1.scene.remove_geometry('lidar cloud')
        except:
            pass

        for i in range(100):
            try:
                self.scene1.scene.remove_geometry(f'bbx object {i}')
            except:
                continue

        self.progress.value = (self.slider.int_value + 1) / len(self.img_files)
        self.progress_text.text = f'({self.slider.int_value}/{len(self.img_files) - 1})'
        self.intedit_h.int_value = self.slider.int_value

        lidar_box_set = self._get_lidar_img(frame_i=frame_i)

        # point cloud
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        # mat.base_color = [0.9, 0.5, 0.5, 0.5]
        self.scene1.scene.add_geometry(f'lidar cloud', self.cloud, mat)

        mat1 = o3d.visualization.rendering.MaterialRecord()
        mat1.shader = "unlitLine"
        mat1.line_width = 2  # note that this is scaled with respect to pixels,
        for li, bbx in enumerate(lidar_box_set):
            self.scene1.scene.add_geometry(f'bbx object {li}', bbx, mat1)

        # image
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.image_widget.update_image(o3d.geometry.Image(self.img))

    def _update_thread(self):

        while not self.is_done:
            time.sleep(0.15)

            if self.play:
                self.slider.int_value += 1

            frame_i = self.slider.int_value

            if self.is_done:
                print(f'break the process')
                break
            else:
                try:
                    if frame_i == 0:  # first frame
                        self.scene1.setup_camera(60, self.scene1.scene.bounding_box, (0, 0, 0))
                    # else:
                    self.app.post_to_main_thread(self.window, lambda: self._update_next_frame(frame_i))
                except Exception as e:
                    print(f'Exit because of: {e}')
                    break

    def show_dataset_o3d(self, folder_i):
        """
        show the example
        """
        print(f'load folder {self.subfolders[folder_i]}')

        self._load_folder(folder_i=folder_i)

        # create open3d window
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Lidar Point Cloud and Camera Image Visualization", 1700, 740)
        em = self.window.theme.font_size

        self.scene1 = gui.SceneWidget()
        self.scene1.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)

        self.scene2 = gui.Vert(0 * 0, gui.Margins(0, 0, 0, 0))
        scene2_collaps = gui.CollapsableVert("Image", 0, gui.Margins(0 * 0, 0 * 0, 0 * 0, 0 * 0))
        self.image_widget = gui.ImageWidget()
        scene2_collaps.add_child(self.image_widget)
        self.scene2.add_child(scene2_collaps)

        self.scene3 = gui.Vert(0 * 0, gui.Margins(0, 0, 0, 0))
        scene3_collaps = gui.CollapsableVert("Control Panel", 0, gui.Margins(0 * 0, 0 * 0, 0 * 0, 0 * 0))
        self.scene3.add_child(scene3_collaps)

        progress_grid = gui.Horiz()
        self.progress = gui.ProgressBar()
        self.progress.value = (0 + 1) / len(self.img_files)  # range(0.0-1.0)
        progress_grid.add_child(self.progress)
        self.progress_text = gui.Label(f'({0}/{len(self.img_files) - 1})')
        progress_grid.add_child(self.progress_text)
        self.scene3.add_child(progress_grid)

        slider_grid = gui.Vert(2)
        slider_grid.add_child(gui.Label('Frame increase'))

        slider_textedit = gui.Horiz()

        self.slider = gui.Slider(gui.Slider.INT)
        slider_max_value = len(self.img_files) - 1
        self.slider.set_limits(0, slider_max_value)
        self.slider.int_value = 0
        self.slider.set_on_value_changed(self._slider_value_changed)

        self.intedit_h = gui.NumberEdit(gui.NumberEdit.INT)
        self.intedit_h.int_value = 0
        self.intedit_h.set_limits(0, len(self.img_files) - 1)
        self.intedit_h.set_on_value_changed(self._intedit_value_changed)

        slider_textedit.add_child(self.slider)
        slider_textedit.add_child(self.intedit_h)

        slider_grid.add_child(slider_textedit)

        button_grid = gui.Horiz(2, gui.Margins(em, em, 0, 0))
        play_button = gui.Button('Play')
        play_button.set_on_clicked(self._play_btn_clicked)
        pause_button = gui.Button('Pause')
        pause_button.set_on_clicked(self._pause_btn_clicked)

        button_grid.add_child(play_button)
        button_grid.add_child(pause_button)

        self.scene3.add_child(slider_grid)
        self.scene3.add_child(button_grid)

        self.window.add_child(self.scene1)
        self.window.add_child(self.scene2)
        self.window.add_child(self.scene3)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_main_window_closing)
        self.is_done = False

        threading.Thread(target=self._update_thread).start()

        self.app.run()


if __name__ == '__main__':
    vis_dataset = VisDataset(vis=True)
    vis_dataset.main()
