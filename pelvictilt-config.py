_base_ = [
    '/data1/data/vscode-ml/posedetect/mmpose/td-hm_resnetv1d152_8xb32-210e_coco-256x192.py',
    '../posedetect/mmpose/configs/_base_/datasets/pelvictilt-custom.py',
]
data_root='/data1/data/vscode-ml/posedetect/確認使用/'
# 模型設置
model = dict(
   backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data1/data/vscode-ml/posedetect/mmpose/td-hm_resnetv1d152_8xb32-210e_coco-256x192-fd49f947_20221021.pth',
            prefix='backbone',
        )),
        head=dict(
        decoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        in_channels=2048,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=7,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True),
    type='TopdownPoseEstimator')

# data loaders
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/train_output.json',        
        data_prefix=dict(img='train_images/'),
        metainfo=dict(
            from_file='/data1/data/vscode-ml/posedetect/mmpose/configs/_base_/datasets/pelvictilt-custom.py'
        )
    )
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/val_output.json',
        bbox_file='/data1/data/vscode-ml/posedetect/確認使用/person_detection_results/val_bbox_output.json',
        data_prefix=dict(img='val_images/'),
        metainfo=dict(
            from_file='/data1/data/vscode-ml/posedetect/mmpose/configs/_base_/datasets/pelvictilt-custom.py'
        ),
        test_mode=True,
    )
)

test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/test_output.json',
        bbox_file='/data1/data/vscode-ml/posedetect/確認使用/person_detection_results/test_bbox_output.json',
        data_prefix=dict(img='test_images/'),
        metainfo=dict(
            from_file='/data1/data/vscode-ml/posedetect/mmpose/configs/_base_/datasets/pelvictilt-custom.py'
        ),
        test_mode=True,
    )
)
val_evaluator = dict(
    ann_file='/data1/data/vscode-ml/posedetect/確認使用/annotations/val_output.json',
    type='CocoMetric')
test_evaluator = dict(
    ann_file='/data1/data/vscode-ml/posedetect/確認使用/annotations/test_output.json',
    type='CocoMetric')

# 优化器超参数
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率策略
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)]

auto_scale_lr = dict(
    enable=True,
    base_batch_size=32
)
