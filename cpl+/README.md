# Weakly and Single-Frame Supervised Temporal Sentence Grounding With Gaussian-Based Contrastive Proposal Learning

In this paper, we extend our CPL to support both weakly and single-frame supervised temporal sentence grounding.

Our paper was accepted by TPAMI-2025. [Paper](https://www.computer.org/csdl/journal/tp/5555/01/11301624/2cwBH2tytUY)

## Main Results

### Charades-STA Dataset

| Method  | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 |
| :-----: | :-------: | :-------: | :-------: |
|   CPL   |   68.49   |   51.83   |   23.84   |

Our trained model can be downloaded from [here](checkpoints/charades/model-best.pt)

### ActivityNet Captions Dataset

| Method  | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 |
| :-----: | :-------: | :-------: | :-------: |
|   CPL   |   85.37   |   62.57   |  37.11    |

Our trained model can be downloaded from [here](checkpoints/activitynet/model-best.pt)

## Requiments

- pytorch
- h5py
- nltk
- fairseq

## Quick Start

### Data Preparation

We use the C3D feature for the ActivityNet Captions dataset. Please download from [here](http://activity-net.org/challenges/2016/download.html) and save as `data/activitynet/c3d_features.hdf5`. We use the I3D feature provided by [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding) and use [this script](/data/convert_npy_to_hdf5.py) to convert the file format to HDF5. We also provide the converted I3D feature for the Charades-STA dataset, and can be downloaded from [here](https://pan.baidu.com/s/1WhWreaHIx8pI5hLK2uyCdw?pwd=4g9h). We expect the directory structure to be the following:

```
data
├── activitynet
│   ├── sub_activitynet_v1-3.c3d.hdf5
│   ├── glove.pkl
│   ├── train_data.json
│   ├── val_data.json
│   ├── test_data.json
├── charades
│   ├── i3d_features.hdf5
│   ├── glove.pkl
│   ├── train.json
│   ├── test.json
```

### Training

To train on the ActivityNet Captions dataset:
```bash
python train.py --config-path config/activitynet/main.json --log_dir LOG_DIR --tag TAG
```

To train on the Charades-STA dataset:
```bash
python train.py --config-path config/charades/main.json --log_dir LOG_DIR --tag TAG
```

Use `--log_dir` to specify the directory where the logs are saved, and use `--tag` to identify each experiment. They are both optional.

The model weights are saved in `checkpoints/` by default and can be modified in the configuration file.

### Inference

Our trained model are provided in [`checkpoints/`](/checkpoints/). Run the following commands for evaluation:

```bash
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval
```

The configuration file is the same as training.
