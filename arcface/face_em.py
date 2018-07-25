from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA

from .utils import preprocess


def get_model(ctx, image_size, model, layer):
    prefix = model
    epoch = 0
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model

class FaceModel:
    def __init__(self, 
                 image_size='112,112',
                 model='./model/model-r50-am-lfw',
                 ctx=mx.cpu()):
        _vec = image_size.split(',')
        assert len(_vec)==2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(model)>0:
          self.model = get_model(ctx, image_size, model, 'fc1')

    def get_input(self, face_img, bbox, points):
        bbox = bbox[0:4]
        points = points[:].reshape((2,5)).T
        nimg = preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        return nimg, aligned
    
    def get_feature(self, aligned):
        #face_img is bgr image
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding