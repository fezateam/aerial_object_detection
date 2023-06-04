dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 466), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 466),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/akay/mm/mmdetection-2.21.0/data/visdrone/train/visdronetrain.json',
        img_prefix=
        '/home/akay/mm/mmdetection-2.21.0/data/visdrone/train/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1280, 466), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/akay/mm/mmdetection-2.21.0/data/visdrone/val/visdroneval.json',
        img_prefix='/home/akay/mm/mmdetection-2.21.0/data/visdrone/val/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 466),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/akay/mm/mmdetection-2.21.0/data/random2022v2/random2022v2.json',
        img_prefix='/home/akay/mm/mmdetection-2.21.0/data/random2022v2/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1280, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                 'tricycle', 'awning-tricycle', 'bus', 'motor')))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=45)
checkpoint_config = dict(interval=-1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='TOOD',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=384,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        num_classes=10,
        in_channels=384,
        stacked_convs=6,
        feat_channels=384,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=2,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.07,
        nms=dict(type='soft_nms', iou_threshold=0.6),
        max_per_img=100))
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
           'tricycle', 'awning-tricycle', 'bus', 'motor')
work_dir = './work_dirs/visdrone_cnext_fpn_thead'
auto_resume = False
gpu_ids = [0]
