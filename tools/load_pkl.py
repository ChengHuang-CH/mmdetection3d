import mmcv

train_data = mmcv.load('../data/nuscenes/nuscenes_infos_train.pkl', file_format='pkl')
valid_data = mmcv.load('../data/nuscenes/nuscenes_infos_val.pkl', file_format='pkl')
print()