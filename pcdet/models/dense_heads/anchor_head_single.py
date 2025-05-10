import time

import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from pcdet.models.model_utils.basic_block_2d import build_block


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.voxel_size = voxel_size
        self.feature_map_stride = model_cfg.ANCHOR_GENERATOR_CONFIG[0]['feature_map_stride']

        if self.model_cfg.get('VOXEL_SIZE', None):
            self.voxel_size = model_cfg.VOXEL_SIZE

        # build pre block
        if self.model_cfg.get('PRE_BLOCK', None):
            pre_block = []

            block_types = self.model_cfg.PRE_BLOCK.BLOCK_TYPE
            num_filters = self.model_cfg.PRE_BLOCK.NUM_FILTERS
            layer_strides = self.model_cfg.PRE_BLOCK.LAYER_STRIDES
            kernel_sizes = self.model_cfg.PRE_BLOCK.KERNEL_SIZES
            paddings = self.model_cfg.PRE_BLOCK.PADDINGS
            in_channels = input_channels

            for i in range(len(num_filters)):
                pre_block.extend(build_block(
                    block_types[i], in_channels, num_filters[i], kernel_size=kernel_sizes[i],
                    stride=layer_strides[i], padding=paddings[i], bias=False
                ))
                in_channels = num_filters[i]
            self.pre_block = nn.Sequential(*pre_block)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_IOU', None) is not None:
            self.conv_iou = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )
        else:
            self.conv_iou = None

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()


    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def forward(self, data_dict):
        try:
            spatial_features_2d = data_dict['spatial_features_2d']
        except:
            spatial_features_2d = data_dict['st_features_2d']
        in_feature_2d = spatial_features_2d

        if self.model_cfg.get('VOXEL_SIZE', None):
            output_size = [self.grid_size[0], self.grid_size[1]]
            in_feature_2d = nn.functional.interpolate(
                in_feature_2d, output_size, mode='bilinear', align_corners=False
            )

        if hasattr(self, 'pre_block'):
            in_feature_2d = self.pre_block(in_feature_2d)
            data_dict['spatial_features_2d_preblock'] = in_feature_2d

        cls_preds = self.conv_cls(in_feature_2d)
        box_preds = self.conv_box(in_feature_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(in_feature_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.conv_iou is not None:
            iou_cls_preds = self.conv_iou(in_feature_2d)
            iou_cls_preds = iou_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['iou_cls_preds'] = iou_cls_preds
        else:
            iou_cls_preds = None

        if self.training:
            target_boxes = data_dict['gt_boxes']


            # visualization code
            # num_gt_boxes = target_boxes.shape[1]

            # label assign kd
            if self.kd_head is not None and not self.is_teacher and self.model_cfg.get('LABEL_ASSIGN_KD', None):
                # import ipdb; ipdb.set_trace(context=20)
                # from pcdet.datasets.dataset import DatasetTemplate
                # DatasetTemplate.__vis_open3d__(
                #     data_dict['points'][:, 1:].cpu().numpy(), target_boxes[0, :, :7].cpu().numpy(),
                #     data_dict['decoded_pred_tea'][0]['pred_boxes'][:, :7].cpu().numpy()
                # )
                target_boxes, num_target_boxes_list = self.kd_head.parse_teacher_pred_to_targets(
                    kd_cfg=self.model_cfg.LABEL_ASSIGN_KD, pred_boxes_tea=data_dict['decoded_pred_tea'],
                    gt_boxes=target_boxes
                )

                # import ipdb; ipdb.set_trace(context=20)
                # from pcdet.datasets.dataset import DatasetTemplate
                # num_target_boxes = int(target_boxes.shape[1] - num_gt_boxes)
                # DatasetTemplate.__vis_open3d__(
                #     data_dict['points'][:, 1:].cpu().numpy(), target_boxes[0, num_target_boxes:, :7].cpu().numpy(),
                #     target_boxes[0, :num_target_boxes, :7].cpu().numpy()
                # )
            # exit()
            targets_dict = self.assign_targets(
                gt_boxes=target_boxes
            )
            # print(target_boxes.shape)
            # print(target_boxes[:,:,-1])
            # exit()

            self.forward_ret_dict.update(targets_dict)
            if not self.is_teacher:
                data_dict['cls_preds'] = self.forward_ret_dict['cls_preds']
                data_dict['box_preds'] = self.forward_ret_dict['box_preds']
                data_dict['iou_preds'] = iou_cls_preds

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

            data_dict['batch_iou_preds'] = iou_cls_preds

            # print('a', box_preds.shape)
            # print('b', batch_box_preds.shape)
            # exit()

            if self.is_teacher:
                data_dict['batch_cls_preds_tea_densehead'] = batch_cls_preds
                data_dict['batch_box_preds_tea_densehead'] = batch_box_preds
                data_dict['cls_preds_normalized_tea_densehead'] = False

            data_dict['cls_preds_vina'] = cls_preds

        return data_dict

import torch
class AnchorHeadSingleCas(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()


    def forward(self, data_dict):

        # anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)

        # new_anchors = []
        # for anchors in self.anchors_root:
        #     new_anchors.append(anchors[:, anchor_mask, ...])

        # self.anchors = new_anchors

        for i in range(1):
            if i==0:
                st_features_2d = data_dict['st_features_2d']

                cls_preds = self.conv_cls(st_features_2d)
                box_preds = self.conv_box(st_features_2d)

                cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

                self.forward_ret_dict['cls_preds'] = cls_preds
                self.forward_ret_dict['box_preds'] = box_preds

                if self.conv_dir_cls is not None:
                    dir_cls_preds = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
                else:
                    dir_cls_preds = None

            else:
                if 'st_features_2d'+str(-i) not in data_dict:
                    continue
                st_features_2d = data_dict['st_features_2d'+str(-i)]

                cls_preds2 = self.conv_cls(st_features_2d)
                box_preds2 = self.conv_box(st_features_2d)


                cls_preds2 = cls_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                box_preds2 = box_preds2.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


                self.forward_ret_dict['cls_preds'+str(-i)] = cls_preds2
                self.forward_ret_dict['box_preds'+str(-i)] = box_preds2

                if self.conv_dir_cls is not None:
                    dir_cls_preds2 = self.conv_dir_cls(st_features_2d)
                    dir_cls_preds2 = dir_cls_preds2.permute(0, 2, 3, 1).contiguous()
                    self.forward_ret_dict['dir_cls_preds'+str(-i)] = dir_cls_preds2
                else:
                    dir_cls_preds2 = None


        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        return data_dict
