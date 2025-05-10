# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation
Please follow the OpenPCDet [tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to 
prepare needed datasets.

## Training & Testing

### Step 1: Train the models (Fade as examples, on 4 GPUs)
```
cd tools/
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh --cfg_file cfgs/kitti_models/fade.yaml --extra_tag baseline
```




### Step 2: Distillation (voxelrcnn to Fade as example, on 4 GPUs) 
```
cd tools/
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --cfg_file cfgs/kitti_cakdp/fade_voxelrcnn.yaml --extra_tag voxelrcnn-fade
```

### Step 3: Testing 
Please modify following "model" first.

```
cd tools/
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_test.sh --cfg_file cfgs/kitti_models/fade.yaml --ckpt {#model}
```

