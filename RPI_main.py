#!/usr/bin/python3

#   READ CAREFULLY: This is an Ondevice server which serves as capturing camera image, classifying and 
#   returning results, And also to provide a web interface to make it easier to interface with.

from flask import *
from flask_restful import *
from flask_sessionstore import Session
import requests
from werkzeug.utils import secure_filename

from json import dumps
import json
import subprocess
import hashlib
import string 
import time 
import os 
import random
from bson import ObjectId

import numpy as np
import matplotlib.pyplot as plt

from picamera.array import PiRGBArray
from picamera import PiCamera

from AIengine import *

DEVICE_ID = "Testpi"

app = Flask(__name__)
app.config.update(
    DATABASE = 'Smart'
)
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = os.urandom(32)#bytes(str(hex(random.getrandbits(128))), 'ascii')
app.config['UPLOAD_FOLDER'] = './models/'

#####################################################################################################

ai_engine = AIengine("./models")

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

#####################################################################################################

def validateUser(uid, upass):
    if uid == 'pi' and upass == 'smart':
        return "normaluser" 
    elif uid == 'admin' and upass == 'smart':
        return "admin" 
    return False

@app.errorhandler(404)
def page_not_found(e):
    return render_template("/404.html")

@app.route("/", methods=["GET", "POST"])        # Home Page
@app.route("/home", methods=["GET", "POST"])    # Future Home Page
def home():
    if "login" in session:
        return redirect("/dashboard")
    else:
        return redirect("/login")#render_template('/homes.html')


@app.route("/login", methods=["GET", "POST"])
@app.route("/login_user", methods=["GET", "POST"])
def login_user():
    if "login" in session:
        return redirect("/dashboard")
    elif request.method == "POST":
        try:
            uid = request.form['uname']
            upass = request.form['pass']
            if(uid == '' or upass == ''):
                return render_template('/login.html')
            val = validateUser(uid, upass)
            if val:
                session["login"] = uid
                session["feedpos"] = 0
                session["type"] = val
                #session["database"] = Database("http://admin:ashish@localhost:5984")
                if val == "admin":
                    return redirect("/admin")
                else:
                    return redirect("/dashboard")
            else:
                return "Incorrect Username/Password"
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    return render_template('/login.html')

########################################## ADMIN Dashboard and Secret Stuffs ##########################################

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if ("login" in session) and ("admin" in session["type"]):
        print("HLP")
        try:
            return redirect("/dashboard")#render_template('admin.html')
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    return render_template('/404.html')


############################################ Dashboard and internal stuffs ############################################


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "login" in session: 
        if request.method == "POST": # Redeem Flags
            #pp = request.form[]
            return render_template('/internal/home.html')
        ss = session['login'] 
        return render_template('/internal/home.html')
    else:
        return redirect("/login_user")
    return render_template('/500.html')


@app.route("/update", methods=["GET", "POST"])
def update():
    # Updates the local model using a file provided
    if "login" in session:
        if request.method == "POST":
            f = request.files['file']
            f.save("model.pkl")
            return jsonify("Success")
        else: 
            return jsonify("You are at the wrong place at the right time, my poor friend")
    else:
        return redirect("/login")
    return render_template('/500.html')


@app.route("/camera", methods=["GET", "POST"])
def camera():
    # Updates the local model using a file provided
    if "login" in session:
            frame = camera.capture(rawCapture, format="bgr", use_video_port=True)
            plt.imshow(frame)
            return jsonify(frame)
    else:
        return redirect("/login")
    return render_template('/500.html')

@app.route("/processedFeed", methods=["GET", "POST"])
def processedFeed():
    # Updates the local model using a file provided
    if "login" in session:
        if request.method == "POST":
            frame = camera.capture(rawCapture, format="bgr", use_video_port=True)
            plt.imshow(frame)
            return jsonify(frame)
        else: 
            return jsonify("You are at the wrong place at the right time, my poor friend")
    else:
        return redirect("/login")
    return render_template('/500.html')



@app.route("/template", methods=["GET", "POST"])
def template():
    # Updates the local model using a file provided
    if "login" in session:
        if request.method == "POST":
            return "Fuck off"
        else: 
            return jsonify("You are at the wrong place at the right time, my poor friend")
    else:
        return redirect("/login")
    return render_template('/500.html')