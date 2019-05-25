import cv2
import pickle
import json
import numpy as np
import argparse

import os 
import subprocess.getoutput

from camera import *

def get_args():
    parser = argparse.ArgumentParser( prog="EagleClient.py",
                        formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50),
                        epilog= '''
                        This is a simple script to capture and generate datasets as well as generate the model
                        ''')
    parser.add_argument("url", default=broker, help="URL/IP for the backend model generator")
    parser.add_argument("-c", "--camera", default=1, help="Camera Index")
    #parser.add_argument("-url", "--port", default=broker_port, help="Broker connection port")
    args = parser.parse_args()
    return args

args = get_args()

q = 'y'
print("Camera index: " + str(args.camera))

global camera
camera = Camera(index=int(args.camera))

print("Welcome to dataset and model generator. Just keep inserting labels and click photos...")

path = input("Insert the root where images need to be stored")
if "No such file or directory" in subprocess.getoutput("ls " + path):
    subprocess.getoutput("mkdir " + path)

imgs = list()
labels = list()

while q != 'n':
    name = input("\nEnter the name/ID of the subject: ")
    n = int(input("How many photos to click? "))
    if "No such file or directory" in subprocess.getoutput("ls " + path + '\\' + name):
        subprocess.getoutput("mkdir " + path)
    for i in range(0, n):
        print("\t\tPress 'q' to capture the photo. ")
        while True:
            img = camera.getFrame()
            imgs.append(np.array(img).tolist())
            labels.append(name)
            cv2.imshow('Frame', img)
            #Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.imwrite(name + "." + str(i) + ".png")
        print("\tImage saved")
    q = input("Do you want exit? (y/n) ")

q = input("Do you want to build a model now? (y/n) ")
if q == 'n':
    exit()

import requests as re 

result = re.post(url, data = json.dumps({'data':imgs, 'labels':labels, 'user':'admin', 'type':'img', 'preprocess':True}))

print(result)

print("\n\n Goodbye!")