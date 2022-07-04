# CoText

## Real-Time End-to-End Video Text Spotting with Contrastive Representation Learning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](pipeline.png)



## Introduction
[Real-Time End-to-End Video Text Spotting with Contrastive Representation Learning]
Video text spotting(VTS) is the task that requires simultaneously detecting, tracking and recognizing text instances in the video. Existing video text spotting methods typically develop sophisticated pipelines and multiple models, which is no friend for real-time applications. Here we propose a real-time end-to-end video text spotter with Contrastive Representation learning (CoText). Our contributions are four-fold: 1) For the first time, we simultaneously address the three tasks (e.g., text detection, tracking, recognition) in a real-time end-to-end trainable framework.
2) Like humans, CoText tracks and recognizes texts by comprehending them, relating them each other with visual and semantic representations. 3) With contrastive learning, CoText models long-range dependencies and learning temporal information across multiple frames. 4) A simple, light-weight architecture is designed for effective and accurate performance, including GPU-parallel detection post-processing, CTCbased recognition head with Masked RoI, and track head with contrastive learning. Extensive experiments show the superiority of our method. Especially, CoText achieves an video text spotting IDF1 of 72.0% at 35.2 FPS on ICDAR2015video [13], with 10.5% and 26.2 FPS improvement
the previous best method.

Link to our new benchmark [BOVText: A Large-Scale, Bilingual Open World Dataset for Video Text Spotting](https://github.com/weijiawu/BOVText-Benchmark)


## Updates

- (04/07/2022) CoText is accepted by ECCV2022.

- (03/31/2022) Refactoring the code.  


## Performance

### [ICDAR2015(video) Tracking challenge](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=1)

Methods | MOTA | MOTP | IDF1 | Mostly Matched |	Partially Matched |	Mostly Lost
:---:|:---:|:---:|:---:|:---:|:---:|:---:
CoText | -	|-|-	|-	|-	|-


### [ICDAR2015(video) Video Text Spotting challenge](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=1)
Methods | MOTA | MOTP | IDF1 | Mostly Matched |	Partially Matched |	Mostly Lost
:---:|:---:|:---:|:---:|:---:|:---:|:---:
CoText | -	|-|-	|-	|-	|-

#### Notes
- The training time is on 8 NVIDIA V100 GPUs with batchsize 16.
- We use the models pre-trained on COCOTextV2.
- We do not release the recognition code due to the company's regulations.


## Demo
<img src="demo.gif" width="400"/>  <img src="demo1.gif" width="400"/>


## Installation
The codebases are built on top of [PAN++](https://github.com/whai362/pan_pp.pytorch).

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n CoText python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate CoText
    ```
  
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build KP
    ```bash
	cd models/kp
	python setup.py clean && python setup.py bdist_wheel
	cd dist && pip install kprocess-0.1.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
    ```
## Usage

### Dataset preparation

1. Please download [ICDAR2015](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=4) and [COCOTextV2 dataset](https://bgshih.github.io/cocotext/) and organize them like [FairMOT](https://github.com/ifzhang/FairMOT) as following:


2. You also can use the following script to generate txt file:


```bash 
cd ../../
```
### Training and Evaluation

#### Training on single node

You can download COCOTextV2 pretrained weights from Pretrained CoText. Or training by youself:
```bash 
sh configs/r50_TransDETR_pretrain_COCOText.sh

```

Then training on ICDAR2015 with 8 GPUs as following:

```bash 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py config/pan_pp/pan_pp_r18_coco_detrec.py 

```

#### Evaluation on ICDAR13 and ICDAR15

You can download the pretrained model of CoText (the link is in "Main Results" session), then run following command to evaluate it on ICDAR2015 dataset:

```bash 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py config/pan_pp/pan_pp_r18_ic15_desc.py

```

#### Visualization 

```bash 
python infer.py # 单图检测、识别、desc
python track_icd15.py

```



## License

CoText is released under MIT License.


## Citing

If you use CoText in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@article{wu2022cotext,
  title={Real-time End-to-End Video Text Spotter with Contrastive Representation Learning},
  author={Weijia Wu, Zhuang Li, Jiahong Li, Chunhua Shen, Hong Zhou, Size Li, Zhongyuan Wang, Ping Luo},
  journal={ECCV2022},
  year={2022}
}
```
