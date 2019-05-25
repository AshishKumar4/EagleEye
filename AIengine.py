
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import sklearn
import re
from scipy.spatial import distance
from skimage.transform import resize
import subprocess

from sklearn.svm import SVC
from scipy import misc

import cv2
import tensorflow as tf

from mtcnn.mtcnn import MTCNN
import mxnet as mx

np.random.seed(10)

cv2_face_detector = cv2.dnn.readNetFromCaffe('./cv2/deploy.prototxt.txt', './cv2/res10_300x300_ssd_iter_140000.caffemodel')

def face_extract_dnn(img, margin=0, image_size=160, model=cv2_face_detector):
    try:
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (image_size, image_size)), 1.0, (image_size, image_size), (103.93, 116.77, 123.68))
        model.setInput(blob)
        detections = model.forward()
        confidence = detections[0, 0, 0, 2]
        if confidence >= 0.55:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (x, y, p, q) = box.astype('int')
            (y1, y2, x1, x2) = (y-margin//2,q+margin//2, x-margin//2,p+margin//2)
            (x1, y1) = [0 if i<0 else i for i in (x1, y1)]
            faceDetected = (x1, y1, x2, y2) 
            cropped = img[y1:y2, x1:x2, :]
            _img = resize(cropped, (image_size, image_size),
                          mode='reflect', anti_aliasing=True)
            return _img, faceDetected
        else:
            return img, False
    except Exception as e:
        print("Error in face_extract_dnn")
        print(e)
        print("\t\t" + str(confidence) + " " + str(faceDetected))
        return img, False
    return None

def face_extract_mtcnn(img, margin=0, image_size=160, model=MTCNN()):
    try:
        detections = model.detect_faces(img)
        confidence = [i['confidence'] for i in detections][0]
        if confidence >= 0.55:
            box = detections[0]['box']
            (x, y, p, q) = box
            (y1, y2, x1, x2) = (y-margin//2,q+y+margin//2, x-margin//2,p+x+margin//2)
            (x1, y1) = [0 if i<0 else i for i in (x1, y1)]
            faceDetected = (x1, y1, x2, y2) 
            cropped = img[y1:y2, x1:x2, :]
            _img = resize(cropped, (image_size, image_size),
                          mode='reflect')
            return _img, faceDetected
        else:
            return img, False
    except Exception as e:
        print("Error in face_extract_dnn")
        print(e)
        print("\t\t" + str(confidence) + " " + str(faceDetected))
        return img, False
    return None

def face_extract_haar(img, margin=70, image_size=160, cascade_path='./cv2/haarcascade_frontalface_alt2.xml'):
    cascade = cv2.CascadeClassifier(cascade_path)
    try:
        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        faceDetected = (x, y, w, h)
        # print(faces[0].dtype)
        cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
        img = resize(cropped, (image_size, image_size), mode='reflect')
    except Exception as e:
        print("error in face detection")
        print(e)
        img = resize(img, (image_size, image_size), mode='reflect')
        faceDetected = False
    return img, faceDetected


if tf.__version__ == '2.0.0-alpha0':
    coreModel = tf.keras.models.load_model("./models/facenet_512.h5")
else:
    import keras
    coreModel = keras.models.load_model(
        "./models/facenet_512_tf1.h5", custom_objects={'tf': tf})

def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
  

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                                           axis=axis, keepdims=True), epsilon))
    return output

def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])


def to_rgb(img):
    if img.ndim == 2:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=img.dtype)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    elif img.shape[2] == 3:
        return img
    elif img.shape[2] == 4:
        w, h, t = img.shape
        ret = np.empty((w, h, 3), dtype=img.dtype)
        # ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return img[:, :, :3]

class AIengine:
    def __init__(self, modelpath='./models', create=False):
        try:
            self.modelpath = modelpath
            classifier = modelpath + "/model.pkl"
            meta = modelpath + "/model.meta"
            self.clfMap = {'img': self.classifyImg, 'vec': self.classifyVec}
            self.fitMap = {'img': self.fitImg, 'vec': self.fitVec}
            print(classifier)

            if create:
                print("Creating new AI Engine")
                #   We Need to create this AI engine and save it on disk
                self.classifier = SVC(kernel='linear', probability=True)

                self.labelEncodeMap = dict()    # Contains mapping from id string to hash
                # self.metadata['labelDecodeMap']   # Contains mapping from hash to id string
                self.labelDecodeMap = dict()

                self.image_size = 160
                self.margin = 1.1

                subprocess.getoutput("mkdir " + modelpath)
                self.save(modelpath)
            else:
                if ("No such file or directory" in (subprocess.getoutput("ls " + modelpath))):
                    return "AI Engine not created yet!"
                print("Loading AI Engine")
                try:
                    self.classifier = pickle.loads(open(classifier, 'rb').read())
                except Exception as e:
                    print(e)
                    print("Seems like the classifier was not found/is corrupt. Would make a new one for you")
                    self.classifier = SVC(kernel='linear', probability=True)
                self.metadata = pickle.loads(open(meta, 'rb').read())

                # Contains mapping from id string to hash
                self.labelEncodeMap = dict(self.metadata['labelEncodeMap'])
                # self.metadata['labelDecodeMap']   # Contains mapping from hash to id string
                self.labelDecodeMap = {value: key for key,
                                       value in self.labelEncodeMap.items()}

                self.image_size = self.metadata['imagesize']  # 160
                self.margin = self.metadata['similarity_margin']  # 1.1

                ctx = mx.gpu(0)
                sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
                all_layers = sym.get_internals()
                sym = all_layers['fc1_output']
                model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
                #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
                model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
                model.set_params(arg_params, aux_params)
                self.model = model
        except Exception as e:
            print("Error in AIengine.init")
            print(e)
        return None
    
    def embed_arcface(self, img):
        nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        #print(nimg.shape)
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

    def embed(self, images, preprocess=False):
        try:
            status = True
            if preprocess is True:
                images, status = self.preprocess(images, 10)
            #emb = l2_normalize(coreModel.predict(images)), status

            return emb
        except Exception as e:
            print("Error in AIengine.embed")
            print(e)
        return None, False

    def fitImg(self, images, labels):
        try:
            embs, status = self.embed(images)
            self.classifier.fit(embs, labels)
            return embs
        except Exception as e:
            print("Error in AIengine.fitImg")
            print(e)
        return False

    def fitVec(self, vectors, labels):
        try:
            embs, status = vectors  # self.embed(images)
            self.classifier.fit(embs, labels)
            return embs
        except Exception as e:
            print("Error in AIengine.fitVec")
            print(e)
        return False

    def fit(self, data, labels, fitType='img'):
        try:
            lbls = list()
            for i in labels:
                if i not in self.labelEncodeMap:
                    self.labelEncodeMap[i] = hash(i)
                    self.labelDecodeMap[hash(i)] = i
                lbls.append(self.labelEncodeMap[i])
            print(self.labelEncodeMap)
            return self.fitMap[fitType](data, lbls).tolist()
        except Exception as e:
            print("Error in AIengine.fit")
            print(e)
        return False

    def classifyImg(self, images, preprocess=True):
        vectors, status = self.embed(images, preprocess)
        val = [self.labelDecodeMap[i] for i in self.classifier.predict(vectors) if i in self.labelDecodeMap]
        if len(val) == 0:
            return None, False 
        return val, status

    def classifyVec(self, vectors, preprocess):
        hashs = self.classifier.predict(vectors)
        val = [self.labelDecodeMap[i] for i in hashs if i in self.labelDecodeMap]
        for i in hashs:
            if i not in self.labelDecodeMap:
                print("hash " + str(i) + " Not in map")
        if val is None or len(val) == 0:
            print("no class predicted")
            return None, False 
        return val, True

    def classify(self, data, clfType='img', preprocess=True):
        try:
            return self.clfMap[clfType](data, preprocess)
        except Exception as e:
            print(clfType)
            print("Error in AIengine.classify")
            print(e)
        return None, False

    def isSimilarII(self, img1, img2, margin=1.1):
        try:
            v1 = self.embed([img1])
            v2 = self.embed([img2])
            dis = distance.euclidean(v1, v2)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarII")
            print(e)
        return None

    def isSimilarIV(self, img, vec, margin=1.1):
        try:
            v1 = self.embed([img])
            v2 = vec
            dis = distance.euclidean(v1, v2)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarIV")
            print(e)
        return None

    def isSimilarVV(self, vec1, vec2, margin=1.0):
        try:
            v1 = vec1
            v2 = vec2
            dis = distance.euclidean(v1, v2)
            print(dis)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarVV")
            print(e)
        return None

    @staticmethod
    def prewhiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    @staticmethod
    def preprocess(images, margin=0, image_size=160, model_path='./cv2/haarcascade_frontalface_alt2.xml', face_extract_algo=face_extract_dnn):
        try:
            faceDetected = True
            aligned_images = []
            for img in images:
                # print(filepath)
                if type(img) is list:
                    img = np.array(img)
                img = to_rgb(img)
                img, faceDetected = face_extract_algo(img, margin)
                if faceDetected is False:
                    # img, faceDetected = AIengine.face_extract_haar(img)
                    # if faceDetected is False:
                    #    continue
                    continue
                aligned_images.append(img)
            if len(aligned_images) == 0:
                return images, False
            return np.array(aligned_images), faceDetected
        except Exception as e:
            print("Error in Preprocess ")
            print(e)
            return images, False

    def save(self, modelpath):
        self.modelpath = modelpath
        self.metadata = {'labelEncodeMap': self.labelEncodeMap, 'labelDecodeMap': self.labelDecodeMap,
                         'imagesize': self.image_size, 'similarity_margin': self.margin}
        f = open(modelpath+'/model.meta', 'wb')
        f.write(pickle.dumps(self.metadata))
        f.close()

        f = open(modelpath+'/model.pkl', 'wb')
        f.write(pickle.dumps(self.classifier))
        f.close()
        return True
