_base_ = './htc_r50_fpn_1x_coco.py'
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=20)
