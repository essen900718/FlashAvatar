## 踩坑日記

1. **pytorch3d install failed**: add `-c conda-forge` when installing pytorch, torchvision, and torchaudio
2. **fail to install diff-gaussian-rasterization and simple-knn submodules**: update cudatoolkit to 11.6 or newer
3. **ImportError: cannot import name 'bool' from 'numpy'**: downgrade numpy to 1.23.1

用官方給的指令實在是裝不起來，折騰了半天最後用另一個[repo](https://github.com/ShenhanQian/GaussianAvatars)的方法馬上就裝好了!

```
conda create -n flashavatar -y python=3.10
conda install ninja
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
pip install torch==2.0.1 torchvision==0.15.2
pip install -r requirements.txt
pip install mxnet-mkl==1.6.0 numpy==1.23.1
```

**Note!!!**: remember to download `generic_model.pkl` and `FLAME_masks.pkl` and place them to `flame/generic_model.pkl` and `flame/FLAME_masks/FLAME_masks.pkl`

**Testing**
```
python test.py --idname Obama --checkpoint dataset/Obama/log/ckpt/chkpnt.pth
```

---

# FlashAvatar
**[Paper](https://arxiv.org/abs/2312.02214)|[Project Page](https://ustc3dv.github.io/FlashAvatar/)**

![teaser](exhibition/teaser.png)
Given a monocular video sequence, our proposed FlashAvatar can reconstruct a high-fidelity digital avatar in minutes which can be animated and rendered over 300FPS at the resolution of 512×512 with an Nvidia RTX 3090.

## Setup

This code has been tested on Nvidia RTX 3090. 

Create the environment:

```
conda env create --file environment.yml
conda activate FlashAvatar
```

Install PyTorch3D:

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```
## Data Convention
The data is organized in the following form：
```
dataset
├── <id1_name>
    ├── alpha # raw alpha prediction
    ├── imgs # extracted video frames
    ├── parsing # semantic segmentation
├── <id2_name>
...
metrical-tracker
├── output
    ├── <id1_name>
        ├── checkpoint
    ├── <id2_name>
...
```
## Running
- **Evaluating pre-trained model**
```shell
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```
-  **Training on your own data** 
```shell
python train.py --idname <id_name>
```
Download the [example](https://drive.google.com/file/d/1_WLvlmHD73jOAO178N7eX5UQqlrL2ghD/view?usp=drive_link) with pre-processed data and pre-trained model for a try!

## Citation
```
@inproceedings{xiang2024flashavatar,
      author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
      title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2024},
  }
```
