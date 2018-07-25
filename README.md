# wider_person_search
WIDER Person Search Challenge

under construction ...

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

Put the pre-trained face model [LResNet50E-IR@BaiduNetdisk](https://pan.baidu.com/s/1mj6X7MK) in `./arcface/model/` and unzip it. Folder structure like:

```Shell
|- arcface
    |- model
        |- model-r50-am-lfw-0000.params
        |- model-r50-am-lfw-symbol.json
    |- __init__.py
    ...
```




