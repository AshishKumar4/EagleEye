import pymongo
from bson.objectid import ObjectId
import hashlib
import json
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

import re

def getAge(date):
    return 19

class Database:
    def __init__(self, url):
        self.client = pymongo.MongoClient(url)
        self.db = self.client['smart']
        return 

    def getUserVectors(self, user):
        d = self.db 
        try:
           # b = d['users'][data['id']]
            uu = d['users'].find_one({'_id':data['uname']})
            uvecs = list(uu['vectors'])
            return uvecs
        except Exception as e:
            print("Error in getUserVectors()")
            print(e)
            return None 
        return None
        
class DataFile:
    def __init__(self, filename):
        # All Data stored in pickle format with user -> (vectors list) mapping
        self.data = pickle.loads(open(filename, 'rb').read())
        return 

    def getUserVectors(self, user):
        try:
            uvecs = list(self.data[user])
            return uvecs
        except Exception as e:
            print("Error in getUserVectors()")
            print(e)
            return None 
        return None