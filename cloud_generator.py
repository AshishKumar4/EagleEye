#!/usr/bin/python3
from flask import *
from flask_restful import *
from json import dumps
import subprocess

from AIengine import *

import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app)

privateModels = dict()

userModelRoot = lambda obj: './models/'# + obj['user']

class Classify(Resource):
    def post(self):
        # Get Data (Image/Vector), type (img,vec), 'user' - username
        obj = request.get_json(force=True)
        #print(obj)
        if obj['user'] not in privateModels:
            # Not yet loaded in memory, load it
            privateModels[obj['user']] = AIengine(userModelRoot(obj))
        images = np.array(obj['data'])
        if obj['preprocess'] == True:
            images, status = AIengine.preprocess(images)
        results, status = privateModels[obj['user']].classify(images, clfType = obj['type'])
        #print(np.array(obj['data']).shape)
        print(results)
        return jsonify(results)

    
class Train(Resource):
    def post(self):
        # Get Data (Image/Vector), labels (Names), type (img,vec), 'user' - username
        obj = request.get_json(force=True)
        #print(obj)
        if obj['user'] not in privateModels:
            # Not yet loaded in memory, load it
            privateModels[obj['user']] = AIengine(userModelRoot(obj))
        images = np.array(obj['data'])
        if obj['preprocess'] == True:
            images, status = AIengine.preprocess(images)
        results, status = privateModels[obj['user']].fit(images, obj['labels'], obj['type'])
        privateModels[obj['user']].save(userModelRoot(obj))
        #print(results)
        return jsonify(results)


class Similarity(Resource):
    def post(self):
        # Get Data (Image1, data), 'user' - username, type = img if data is img, else vec
        obj = request.get_json(force=True)
        #print(obj)
        if obj['user'] not in privateModels:
            # Not yet loaded in memory, load it
            privateModels[obj['user']] = AIengine(userModelRoot(obj))
        img = np.array([obj['img']])
        if obj['preprocess'] == True:
            img, status = AIengine.preprocess(img)

        if obj['type'] == 'img':
            data = np.array([obj['data']])
            if obj['preprocess'] == True:
                data, status = AIengine.preprocess(data)
            results = privateModels[obj['user']].isSimilarII(img, data)
        else :
            results = privateModels[obj['user']].isSimilarIV(np.array([obj['img']]), np.array([obj['data']]))
        privateModels[obj['user']].save(userModelRoot(obj))
        print(results)
        return jsonify(results)

class loadModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']] =  AIengine(userModelRoot(obj))
        return jsonify("Loaded")

class saveModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']].save(userModelRoot(obj))
        return jsonify("Saved")

class createModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']] = AIengine(userModelRoot(obj), create=True)
        return jsonify("Created")

api.add_resource(Classify, '/classify')

api.add_resource(Train, '/train')
api.add_resource(Similarity, '/similarity')

api.add_resource(saveModel, '/save')
api.add_resource(loadModel, '/load')

api.add_resource(createModel, '/create')

# WARNING: DO NOT RUN THIS ON MULTITHREADED WSGI SERVERS!
if __name__ == '__main__':
     app.run(host='0.0.0.0', port='5000')



