
import cv2 
import matplotlib.pyplot as plt
import numpy as np

cameraMap = dict()

gamma = 1.27
invGamma = 1.0 / gamma
gamma_table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

class Camera:
    def __init__(self, camType=None, resolution = (640, 480), index = 0):
        if camType is None:
            try:
                from picamera import PiCamera   # If its a raspberry pi, This would not through any error
                print("Raspberry Pi Pi camera Detected, Loading...")
                self.camera = piCam(resolution)
            except Exception as e:
                print("Webcam detected, Loading...")
                self.camera = webCam(resolution, index)
        else:
            self.camera = cameraMap[camType](resolution)
    
    def getFrame(self):
        frame = self.camera.getFrame()
        frame = cv2.add(frame,np.array([50.0]))#cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame = cv2.LUT(frame, gamma_table)
        return frame

class webCam(Camera):
    def __init__(self, resolution = (640, 480), index = 0):
        self.camera = cv2.VideoCapture(index)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def getFrame(self):
        ret, frame = self.camera.read()
        frame = cv2.flip(frame,1)
        return frame

class piCam(Camera):
    def __init__(self, resolution = (640, 480)):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        self.rawCapture = PiRGBArray(camera, size=(640, 480))
    def getFrame(self):
        frame = camera.capture(rawCapture, format="bgr", use_video_port=True)
        rawCapture.truncate(0)
        return frame

cameraMap['rpi'] = piCam
cameraMap['webcam'] = webCam