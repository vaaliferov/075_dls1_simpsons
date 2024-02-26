import numpy as np
import onnxruntime as rt
from PIL import Image, ImageOps


def pad(im):
    w, h = im.size; m = np.max([w, h])
    hp, hpr = (m - w) // 2, (m - w) % 2
    vp, vpr = (m - h) // 2, (m - h) % 2
    return (hp + hpr, vp + vpr, hp, vp)

def norm(x):
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    return (x - mean) / std

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Model:

    def __init__(self, model_path, labels_path):

       with open(labels_path) as fd:
           self.labels = fd.read().splitlines()
           self.labels = np.array(self.labels)

       self.model = rt.InferenceSession(model_path)

    def predict(self, path, topn=3):

        sz = (224,224)
        im = Image.open(path)
        im.thumbnail(sz, Image.LANCZOS)
        im = ImageOps.expand(im, pad(im))

        x = np.array(im) / 255.
        x = np.float32(norm(x))
        x = x.transpose(2,0,1)
        x = x.reshape((1,) + x.shape)
        y = self.model.run(None, {'x': x})

        probs = softmax(y[0][0])
        idx = np.argsort(-probs)[:topn]
        return self.labels[idx], probs[idx]