# NA-OSTrack

### Noise Aware OSTrack 

Occlusion is one of the most significant challenges encountered by object detectors and trackers. In object tracking, occlusions significantly undermine
the performance of tracking algorithms. 
However, being able to track an object of interest through occlusion has been a long standing challenge for different autonomous tasks.
Here, An occlusion handling algorithm has been developed for OSTrack, one of state-of-the art visual object trackers available. This occlusion handling algorithm, proposed in Atom Tracker, has been further developed and modify into a usable structure for any tracker. The Occlusion Handling algorithm is inside the ``occlusion_handler`` folder and can be used for any tracker that calculates a response map.

<div align="left">
      <a href="https://youtu.be/t-67TLveEvg">
         <img src="https://img.youtube.com/vi/t-67TLveEvg/0.jpg" style="width:65%;">
      </a>
</div>

## Install the environment
All requirements and packages are for OSTrack. 

**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n na_ostrack python=3.8
conda activate na_ostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```

**Option3**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

## How to apply Noise Aware Mechanism to your own SOT tracker? :

Firstly, You should import OcclusionHandler class from ``occlusion_handler/occlusion_handler.py ``. Then create a object and set to initialze values the object. Image is first frame of your sequence and init_bbox is initial values of target bbox [x,y,w,h]. If you want use Kalman Filter to predict state of the object when object lost, You can set the 3rd argument of initialize() to True

```python
occ_handler = OcclusionHandler()
occ_handler.initialize(image, init_bbox)
```

Then, You may call check() function. In this section:

image : Image is frame at time t

prev_state: previous state is input values of your tracker at time t

curr_state: current state is output values of your tracker at time t   

response_hn: response_hn is response map of your tracker at time t (with hanning wındow)

As a result the check() method returns new state [x,y,w,h]. You can feed with new_state to your SOT tracker.

```
new_state = occ_handler.check(image, prev_state, curr_state, response_hn)
```

## Acknowledgments
* Thanks for the [OSTRACK](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
* We use the implementation of the OSTrack from the [OSTRACK](https://github.com/botaoye/OSTrack) repo.  
* Furthermore, we use the implementation of 2-D Kalman Filter from the [Kalman_Filter](https://github.com/RahmadSadli/2-D-Kalman-Filter) repo. 




