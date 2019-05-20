from camera import *
from AIengine import *

import cv2

DEVICE_ID = 0x01
CLASSIFIER_ID = "model1"

camera = Camera()
ai_engine = AIengine('./models')

######################################################################################################################################
########################################################### OUR GENERATORS ###########################################################
######################################################################################################################################

def rawStream():
    while True:
        frame = camera.getFrame()
        ret, encframe = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + encframe.tobytes() + b'\r\n\r\n')

def processedStream():
    while True:
        frame = camera.getFrame()
        pre, status = AIengine.preprocess(np.array([frame]))
        print(status)
        name, detect = ai_engine.classify(pre, preprocess = False)
        name = name[0]
        print(name)
        cv2.putText(frame, name, (10,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        if status is not False:
            # Create bounding box as face has been detected
            (x, y, w, h) = status
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1);
        #cv2.imshow('Frame', pre[0])
        # Press Q on keyboard to  exit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
        ret, encframe = cv2.imencode('.jpg', frame)
        #ret, encface = cv2.imencode('.jpg', pre[0])
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + encframe.tobytes() + b'\r\n\r\n')# + b'Content-Type: image/jpeg\r\n\r\n' + encface.tobytes() + b'\r\n\r\n')


######################################################################################################################################
############################################################ MQTT Servers ############################################################
######################################################################################################################################

