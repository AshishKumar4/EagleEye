import hashlib
import json
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

import re

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