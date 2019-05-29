from align_dlib import *
import numpy as np
import json 
import pickle
from sklearn.svm import SVC
import subprocess

np.random.seed(10)

coreModel = dlib.face_recognition_model_v1('./models/dlib_model.dat')
final_img_size = 160

class dlibEngine:
    def __init__(self, modelpath='./models', create=False):
        try:
            self.modelpath = modelpath
            classifier = modelpath + "/model.pkl"
            meta = modelpath + "/model.meta"
            print(classifier)
            if create:
                print("Creating new AI Dlib Engine")
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
        except Exception as e:
            print("Error in AIengine.init")
            print(e)
        return None

    @staticmethod
    def preprocess(images, margin=10, image_size=160):
        try:
            faceDetected = True
            aligned_images = []
            detections = []
            landmarks = []
            for img in images:
                if type(img) is list:
                    img = np.array(img)
                img = to_rgb(img)
                bb = dlib_model.getLargestFaceBoundingBox(img)
                if bb is None:
                    print("dlib couldn't find any face, using dnn...")
                    #aligned = img
                    #_img, faceDetected = face_extract_dnn(aligned, margin, image_size = image_size)
                    # Comment above lines an uncomment below lines to ignore bad quality face images
                    continue
                else:
                    lm = dlib_model.findLandmarks(img, bb = bb)
                    aligned = dlib_model.align(img, bb=bb, landmarks=lm)
                    if aligned is None:
                        print("Error! No aligned photo")
                        aligned = img
                    x, y, w, h = face_utils.rect_to_bb(bb)
                    faceDetected = (x, y, x + w, y + h)
                if faceDetected is False:
                    print("No face detected")
                    print(type(aligned))
                    continue
                detections.append(bb)
                aligned_images.append(aligned)
                landmarks.append(lm)
            if len(aligned_images) == 0:
                return images, False, None
            return np.array(aligned_images), detections, landmarks 
        except Exception as e:
            print("Error in Preprocess ")
            print(e)
            return images, False, None

    def classify(self, data, clfType='img', preprocess=True):
        try:
            vectors, status = self.embed(data, preprocess)
            val = [self.labelDecodeMap[i] for i in self.classifier.predict(vectors) if i in self.labelDecodeMap]
            if len(val) == 0:
                return None, False 
            return val, status
        except Exception as e:
            print(clfType)
            print("Error in AIengine.classify")
            print(e)
        return None, False

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

    def embed_dlib(images, preprocess=False, landmarks=None):
        try:
            status = True
            if preprocess is True:
                images, status = self.preprocess(images, 10)
            emb = list()
            for i in range(0, len(images)):
                if images[i].dtype != 'uint8':
                    images[i] = (images[i]*255).astype('uint8')
                emb.append(coreModel.compute_face_descriptor(images[i], landmarks[i]))
            return emb, status
        except Exception as e:
            print("Error in AIengine.embed_dlib")
            print(e)
        return None, False