# wider_person_search
WIDER Person Search Challenge



* `arcface`: tools for face feature embedding.  (mxnet)
* `data`: Folder to save images for ReID features extraction
* `featrues`: Folder to save features of face & ReID.
* `mtcnn`: tools for face detection. (mxnet)
* `reid`: tools for global ReID feature embedding. (pytorch)
* `utils`: some utilities.
* `crop.py`: make images for ReID features extraction from original dataset -> `./data`
* `eval.py`: tools for evaluation.
* `face_det_em.py`: face detection & face feature embedding. (validation or test)
* `wider_extract.py`: global ReID feature embedding. (validation and test)
* `rank.py`: get the final result. (validation or test)

## Environment & Dependency

- Python 3.+
- opencv-python: ```pip3 install opencv-python```
- mxnet: ```pip3 install mxnet-cu80```
- pytorch: ```pip3 install torch torchvision visdom```
- numpy, scipy, sklearn, skimage: ```pip3 install numpy scipy scikit-learn scikit-image```


## Prepare Train models

### arcface

Put the pre-trained face model [LResNet50E-IR@BaiduNetdisk](https://pan.baidu.com/s/1mj6X7MK) in `./arcface/model/` and unzip it. Folder structure is like:

```
|- arcface
    |- model
        |- model-r50-am-lfw-0000.params
        |- model-r50-am-lfw-symbol.json
    ...
```

### reid

Put the pre-trained ReID model [ResNet-101@BaiduNetdisk](https://pan.baidu.com/s/1E-ogV3lwlXNGcYSGGqZ1ZQ), [DenseNet-121@BaiduNetdisk](https://pan.baidu.com/s/1X2euVExswFwXkCcrXMWtIQ), [SEResNet-101@BaiduNetdisk](https://pan.baidu.com/s/1MC1GrGt6T8-2wxPwhxFbXw) and [SEResNeXt-101@BaiduNetdisk](https://pan.baidu.com/s/1OJIz5vHVrZA-fOXgF39bLA) in `./reid/models/trained_model/`. Folder structure is like:

```
|- reid
    |- models
        |- trained_models
            |- resnet101_best_model.pth.tar
            |- densenet121_best_model.pth.tar
            |- seresnet101_best_model.pth.tar
            |- seresnext101_best_model.pth.tar
        ...
    ...
```

## Feature extraction

### Face detection & face featrue embedding

1. modify lines 13~14 of `face_det_em.py` to your own data path, for example: 

```Python
trainval_root = '/data2/xieqk/wider/person_search_trainval'
test_root = '/data2/xieqk/wider/person_search_test'
```

Folder structure is like:

```
|- wider
    |- person_search_trainval
        |- train
            |- tt0048545
            ...
        |- val
            |- tt0056923
            ...
        |- train.json
        |- val.json
    |- person_search_test
        |- test
            |- tt0038650
            ...
        |- test.json
```

2. Face detection and face feature extraction (Default gpu_id = 0)

```Shell
# validation set detection & embedding, output: ./features/face_em_val.pkl
python face_det_em.py

# test set detection & embedding, output: ./features/face_em_test.pkl
python face_det_em.py --is-test 1

# change gpu devices: 
# python face_det_em.py --gpu 2
```

### ReID feature extraction

1. Data preparation: crop out all the images of the candidates and name them with their id.

modify lines 6~7 of `crop.py` to your own data path, for example: 

```Python
trainval_root = '/data2/xieqk/wider/person_search_trainval'
test_root = '/data2/xieqk/wider/person_search_test'
```

then run:


```Shell
python crop.py  # the result is in ./data/wider_exfeat/val & ./data/wider_exfeat/test
```

2. feature embedding, run:

```Shell
# use ResNet-101
python wider_extract.py -a resnet101
# use DenseNet-121
python wider_extract.py -a densenet121
# use SEResNet-101
python wider_extract.py -a seresnet101
# use SEResNeXt-101
python wider_extract.py -a seresnext101

# change gpu devices
# python wider_extract.py -a resnet101 --gpu 1
```

### Get the final rank list

After getting all face and ReID features, that means:

```
|- features
    |- face_em_test.pkl     # face features (test set)
    |- face_em_val.pkl      # face features (validation set)
    |- reid_em_test_densenet121.pkl     # DenseNet-121 ReID features (test set)
    |- reid_em_test_resnet101.pkl       # ResNet-101 ReID features (test set)
    |- reid_em_test_seresnet101.pkl     # SEResNet-101 ReID features (test set)
    |- reid_em_test_seresnext101.pkl    # SEResNeXt-101 ReID features (test set)
    |- reid_em_val_densenet121.pkl      # DenseNet-121 ReID features (validation set)
    |- reid_em_val_resnet101.pkl        # ResNet-101 ReID features (validation set)
    |- reid_em_val_seresnet101.pkl      # SEResNet-101 ReID features (validation set)
    |- reid_em_val_seresnext101.pkl     # SEResNeXt-101 ReID features (validation set)
```

just run:

```Shell
# get final rank in validation set & evaluation
python rank.py      # with fusion features

# or
# choices = ['resnet101', 'densenet121', 'seresnet101', 'seresnext101']
# python rank.py -a resnet101     # with ResNet-101 features

# get final rank in test set
python rank.py --is-test 1
```

The output is `./val_rank.txt` or `./test_rank.txt`.

## Results

| ReID Features | mAP (%, validation set) | mAP (%, test set) |
| :-----------: | :---------------------: | :---------------: |
| Resnet-101    | 0.6819 | - |
| DenseNet-121  | 0.6831 | - |
| SEResNet-101  | 0.6972 | - |
| SEResNeXt-101 | 0.7007 | - |
| SEResNeXt-101+ResNet-101 | 0.7081 | - |
| SEResNeXt-101+ResNet-101+DenseNet-121 | 0.7009 | - |
| Resnet-101+DenseNet-121+SEResNet-101+SEResNeXt-101| 0.7132 | - |

## References

[1] Zhang, Kaipeng, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. "Joint face detection and alignment using multitask cascaded convolutional networks." IEEE Signal Processing Letters 23, no. 10 (2016): 1499-1503.

[2] Deng, Jiankang, Jia Guo, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018).

[3] Hermans, Alexander, Lucas Beyer, and Bastian Leibe. "In defense of the triplet loss for person re-identification." arXiv preprint arXiv:1703.07737 (2017).

[4] Szegedy, Christian, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. "Rethinking the inception architecture for computer vision." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818-2826. 2016.

## Github projects:

* [mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
* [insightface](https://github.com/deepinsight/insightface)
* [open-reid](https://github.com/Cysu/open-reid)
* [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)