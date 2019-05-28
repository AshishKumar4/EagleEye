import mxnet as mx
import numpy as np
import cv2
import sklearn
import sklearn.preprocessing

class ArcFace:
    def __init__(self, model_path = 'arcface', image_size = (112, 112)):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        try:
            model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names = None)
            model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        except Exception as e:
            if "Compile with USE_CUDA" in e:
                print("GPU Not found, Using CPU...")
                model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names = None)
                model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def predict(self, images):
        try:
            if images.dtype != 'uint8':
                images = (images*255).astype('uint8')
            else:
                images = images
            emb = list()
            for i in images:
                aligned = np.transpose(i, (2,0,1))
                #print(time.time()-t)
                input_blob = np.expand_dims(aligned, axis=0)
                data = mx.nd.array(input_blob)
                db = mx.io.DataBatch(data=(data,))
                self.model.forward(db, is_train=False)
                embedding = self.model.get_outputs()[0].asnumpy()
                embedding = sklearn.preprocessing.normalize(embedding).flatten()
                emb.append(embedding)
            #print(time.time()-t)
            return np.array(emb)
        except Exception as e:
            print("Error in ArcFace.predict")
            print(input_blob.shape)
            print("\n\n")
            #print(e)
        return None