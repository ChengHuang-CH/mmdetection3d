python3 tools/create_data.py kitti  --root-path data/carla_uam_kitti/ --out-dir data/carla_uam_kitti/ --extra-tag kitti
python3 tools/misc/browse_dataset.py configs/mvxnet/revised_mvx.py --output-dir data/out --task multi_modality-det --online

