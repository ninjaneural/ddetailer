model = dict(type='YOLOV3',
             data_preprocessor=dict(
                 type="DetDataPreprocessor",
                 mean=[0, 0, 0],
                 std=[255.0, 255.0, 255.0],
                 bgr_to_rgb=True,
                 pad_size_divisor=32,
             ),
             backbone=dict(type='Darknet', depth=53, out_indices=(3, 4, 5)),
             neck=dict(type='YOLOV3Neck',
                       num_scales=3,
                       in_channels=[1024, 512, 256],
                       out_channels=[512, 256, 128]),
             bbox_head=dict(type='YOLOV3Head',
                            num_classes=1,
                            in_channels=[512, 256, 128],
                            out_channels=[1024, 512, 256],
                            anchor_generator=dict(type='YOLOAnchorGenerator',
                                                  base_sizes=[[(116, 90),
                                                               (156, 198),
                                                               (373, 326)],
                                                              [(30, 61),
                                                               (62, 45),
                                                               (59, 119)],
                                                              [(10, 13),
                                                               (16, 30),
                                                               (33, 23)]],
                                                  strides=[32, 16, 8]),
                            bbox_coder=dict(type='YOLOBBoxCoder'),
                            featmap_strides=[32, 16, 8]),
             test_cfg=dict(nms_pre=1000,
                           min_bbox_size=0,
                           score_thr=0.05,
                           conf_thr=0.005,
                           nms=dict(type='nms', iou_threshold=0.45),
                           max_per_img=100))
                           
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# In version 3.x, validation and test dataloaders can be configured independently
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader  # The configuration of the testing dataloader is the same as that of the validation dataloader, which is omitted here

