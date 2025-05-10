
CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_cakdp/fade_Cas-V_reg.yaml --batch_size 16 --extra_tag withoutz

CUDA_VISIBLE_DEVICES=2,3 bash scripts/dist_train.sh 2 --cfg_file cfgs/kitti_cakdp/pillarnet_Cas-V_reg.yaml --batch_size 16 --extra_tag withoutz
