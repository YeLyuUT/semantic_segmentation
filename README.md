### [Paper](https://arxiv.org/abs/2102.03099) | [UAVid2020 Benchmark](https://competitions.codalab.org/competitions/25224) <br>

Pytorch implementation of our paper [Bidirectional Multi-scale Attention Networks for Semantic Segmentation of Oblique UAV Imagery](https://arxiv.org/abs/2102.03099).<br>
Our code is based on [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821).[[github]](https://github.com/NVIDIA/semantic-segmentation)<br>

## Installation 

* The code is tested with pytorch 1.3 and python 3.6

## Download Weights

* Create a directory where you can keep large files. Ideally, not in this directory.
```bash
  > mkdir <large_asset_dir>
```

* Update `__C.ASSETS_PATH` in `config.py` to point at that directory

  __C.ASSETS_PATH=<large_asset_dir>

* Download pretrained weights from [google drive](https://drive.google.com/open?id=1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and put into `<large_asset_dir>/seg_weights`

## Download/Prepare Data
Download UAVid data [download](https://uavid.nl/#download), then update `config.py` to set the path:
```python
__C.DATASET.UAVID_DIR = <path_to_uavid>
```

## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.


## Train a model

Train uavid2020, using deeplabV3+ + WRN-38 + bidirectional multi-scale attention with pretrained model
```bash
> python -m runx.runx scripts/train_uavid_deepv3MS_bimsa.yml -i
```

## Run inference on UAVid

```bash
> python -m runx.runx scripts/eval_uavid_deepv3MS_bimsa.yml -i
```

Before running inference, path for the snapshot model needs to be configured in cfg file "eval_uavid_deepv3MS_bimsa.yml",

    snapshot: <path_to_file.pth>

The output inference will be saved in directory:

    logs/eval_uavid_deepv3MS_bimsa/submit

You could just zip the subfolders for online benchmark submission.

Example of blended output is as follows,
![alt text](imgs/blendseq22000900.png "example inference")

### Pretrained Weights
The link to our trained weights with 70.8% mIoU score are as provided from google drive,

    https://drive.google.com/file/d/1jMVOHfHtO-z_eIq9cmA2GTo_MPF02ars/view?usp=sharing

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {Bidirectional Multi-scale Attention Networks for Semantic Segmentation of Oblique UAV Imagery},
  author  = {Ye Lyu and George Vosselman and Gui-Song Xia and Michael Ying Yang},
  journal = {arXiv preprint arXiv:2102.03099},
  year    = {2021}
}
```






