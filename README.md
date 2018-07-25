# wider_person_search
WIDER Person Search Challenge

## under construction ...

* `arcface`: tools for face feature embedding.  (mxnet)
* `featrues`: Folder to save features of face & ReID.
* `mtcnn`: tools for face detection. (mxnet)
* `reid`: tools for global ReID feature embedding. (pytorch)
* `utils`: some utilities.
* `eval.py`: tools for evaluation.
* `face_det_em.py`: face detection & face feature embedding. (validation or test)
* `wider_extract.py`: global ReID feature embedding. (validation and test)
* `rank.py`: get the final result. (validation or test)

## Dependency

- mxnet: ```pip3 install mxnet-cu80```
- pytorch: ```pip3 install torch torchvision```


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

Put the pre-trained ReID model [ResNet-101@BaiduNetdisk](#), [DenseNet-121@BaiduNetdisk](#), [SEResNet-101@BaiduNetdisk](#) and [SEResNeXt-101@BaiduNetdisk](#) in `./reid/models/trained_model/`. Folder structure is like:

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

## Feature embedding

### Face detection & face featrue embedding

1. modify lines 13~14 to your own data path, for example: 

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
# validation set detection & embedding, output:./features/face_em_val.pkl
python face_det_em.py

# test set detection & embedding, output:./features/face_em_val.pkl
python face_det_em.py --is-test 1

# change gpu devices: python face_det_em.py --gpu 2
```



