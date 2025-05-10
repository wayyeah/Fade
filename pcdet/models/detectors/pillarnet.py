from .detector3d_template import Detector3DTemplate

from pcdet.models.kd_heads.anchor_head.anchor_kd_head import AnchorHeadKD
from pcdet.models.kd_heads.center_head.center_kd_head import CenterHeadKD

class PillarNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        if self.dense_head is None and self.dense_head_aux is not None:
            self.dense_head = self.dense_head_aux

        self.kd_head = AnchorHeadKD(self.model_cfg, self.dense_head) if model_cfg.get('KD', None) else None

        self.dense_head.kd_head = self.kd_head
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            if not self.training and str(cur_module)[0:12] == 'KDPointTrans':
                continue
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if self.model_cfg.get('KD_LOSS', None) and self.model_cfg.KD_LOSS.ENABLED:
                kd_loss, tb_dict, disp_dict = self.get_kd_loss(batch_dict, tb_dict, disp_dict)
                loss += kd_loss
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    # def post_processing(self, batch_dict):
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     final_pred_dict = batch_dict['final_box_dicts']
    #     recall_dict = {}
    #     for index in range(batch_size):
    #         pred_boxes = final_pred_dict[index]['pred_boxes']

    #         recall_dict = self.generate_recall_record(
    #             box_preds=pred_boxes,
    #             recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST
    #         )

    #     return final_pred_dict, recall_dict