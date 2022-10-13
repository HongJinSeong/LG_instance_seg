# 사용할 모델 선택
_base_ = '../scnet/scnet_x101_64x4d_fpn_20e_coco.py'


model = dict(
    type='SCNet',
    roi_head=dict(
        _delete_=True,
        type='SCNetRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SCNetMaskHead',
            num_convs=12,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            conv_to_res=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        glbctx_head=dict(
            type='GlobalContextHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_weight=3.0,
            conv_to_res=True),
        feat_relay_head=dict(
            type='FeatureRelayHead',
            in_channels=1024,
            out_conv_channels=256,
            roi_feat_size=7,
            scale_factor=2)))


# 데이터 폴더 설정
data_root = 'dataset/'
classes = ('Normal',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='CopyPaste',
         max_num_pasted=50, 
         bbox_occluded_thr=10, 
         mask_occluded_thr=300, 
         selected=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='RandomShift',
        shift_ratio=0.5, 
        max_shift_px=300, 
        filter_thr_px=1),

    dict(
        type='CutOut',
        n_holes=(0,5), 
        cutout_shape=(32,32), 
        cutout_ratio=None, 
        fill_in=(0, 0, 0)),
    
    dict(
        type='RandomCrop',
        crop_size=(800, 1000),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='FilterAnnotations', min_gt_bbox_wh=(1e-05, 1e-05), by_mask=True),
    dict(
        type='Pad',
        size=(800, 1000),
        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

# validation 평가용
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(1024,1280)],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

#실제 test시에 TTA 적용 
# (800, 1000), (1024, 1280), (1200, 1500), (1440, 1800),(1680, 2100)  ==> 0.613이 최대 (제일좋음 이걸로 가는걸로)
# (800, 1000) 빼면 스코어 더낮아짐
# (1680, 2100) 빼면 스코어 더 낮아짐

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[ (800, 1000), (1024, 1280), (1200, 1500), (1440, 1800), (1680, 2100)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "label(polygon)_train.json",
        img_prefix=data_root + "train/",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True,  with_mask=True)
        ],
        filter_empty_gt=False,
        classes=classes
    ),
    pipeline=train_pipeline)



# 데이터 설정
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=train_dataset,
    val=dict(
        type='CocoDataset',
        pipeline=test_pipeline,
        img_prefix=data_root + "train/",
        classes = classes,
        ann_file=data_root + "label(polygon)_train.json"
),
    test=dict(
        type='CocoDataset',
        pipeline=test_pipeline,
        img_prefix=data_root + "test/",
        classes = classes,
        ann_file=data_root + "test.json"
)
)

evaluation = dict(
    metric=[ 'bbox', 'segm'])

runner = dict(type='EpochBasedRunner', max_epochs=1150)

log_config = dict(
    interval=520,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', by_epoch=True)
    ])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.00005)
## learning rate scheduler 없이 먼저해보고 넣기 ####

lr_config=None

# log 저장 위치
checkpoint_config = dict(
    by_epoch=True, interval=1, save_last=True, max_keep_ckpts=1150)
# 평가 방법
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# 사전 가중치 사용
load_from = 'checkpoint/scnet_x101_64x4d_fpn.pth'

# batch size 설정
auto_scale_lr = dict(enable=False, base_batch_size=1)
