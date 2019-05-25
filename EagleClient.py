from camera import *
from AIengine import *
import cv2
import paho.mqtt.client as mqtt
import pickle
import json
import numpy as np
import argparse
from scipy.spatial import distance
import face_recognition

DEVICE_ID = "1"
CLASSIFIER_ID = "model1"

broker = "iot.eclipse.org"
broker_port = 1883

def get_args():
    parser = argparse.ArgumentParser( prog="EagleClient.py",
                        formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50),
                        epilog= '''
                        This is the client program for Attendence System using Facial Recognition.
                        It uses MQTT for communication, and is meant to be used on specialised single purpose hardware.
                        ''')
    parser.add_argument("broker", default=broker, help="URL/IP for the Broker")
    parser.add_argument("-p", "--port", default=broker_port, help="Broker connection port")
    parser.add_argument("-c", "--camera", default=1, help="Camera Index")
    args = parser.parse_args()
    return args

args = get_args()

print("Camera index: " + str(args.camera))

global camera
camera = Camera(index=int(args.camera))

print("Initializing Camera")
ai_engine = AIengine('./models')
print("Initializing AI Engine")

model_location = './models/model.pkl'

from Database import *
global db  
#db = Database("mongodb://localhost:27017/")
db = DataFile("./models/vectormaps.json")

######################################################################################################################################
########################################################### OUR GENERATORS ###########################################################
#####################################################################################################################################


def rawStream():
    while True:
        frame = camera.getFrame()
        yield(frame)
        #yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + encframe.tobytes() + b'\r\n\r\n')


def processedStream():
    while True:
        frame = camera.getFrame()
        pre, detect = AIengine.preprocess(np.array([frame]))
        if detect is False:
            # No Face detected!
            yield(frame, None, False, None)
            continue
        vec, status = ai_engine.embed(pre, preprocess=False)
        name, status = ai_engine.classifyVec(vec, preprocess=False)
        if name is not None:
            name = name[0]
            print(name)
        else:
            name = "Stranger"
        # Create bounding box as face has been detected
        (x, y, w, h) = detect
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
        yield (frame, vec, status, name)
        #cv2.imshow('Frame', pre[0])
        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
        #ret, encframe = cv2.imencode('.jpg', frame)
        #ret, encface = cv2.imencode('.jpg', pre[0])
        # + b'Content-Type: image/jpeg\r\n\r\n' + encface.tobytes() + b'\r\n\r\n')
        #yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + encframe.tobytes() + b'\r\n\r\n')

def validateSimilarity(uvecs, vec, k = 0.65):
    # Returns true if vec is similar to atleast 'k' fraction of vectors
    frac = round(k*len(uvecs))
    #results = [i for i in face_recognition.compare_faces(uvecs, vec) if i is True]#[np.linalg.norm(vec - i) for i in uvecs if np.linalg.norm(vec - i) <= 0.88]#distance.euclidean(i, vec) <= 0.88]
    #results = [np.linalg.norm(vec - i) for i in uvecs if np.linalg.norm(vec - i) <= 0.80]
    results = [distance.cosine(i, vec) for i in uvecs if distance.cosine(i, vec) <= 0.36]
    #print(results)
    print("Similar to " + str(len(results)) + " Photos of the person out of " + str(len(uvecs)))
    print(results)
    #print([distance.cosine(i, vec) for i in uvecs])
    if len(results) >= frac:
        return True 
    return False


######################################################################################################################################
############################################################ MQTT Globals ############################################################
######################################################################################################################################

topic_attendence = CLASSIFIER_ID + "/attendence"    # Write only for client
#topic_update_model = CLASSIFIER_ID + "/update/model"
#topic_update_database = CLASSIFIER_ID + "/update/database"
topic_control_read = CLASSIFIER_ID + "/control/read"    # Read only for client, would be subscribed
topic_control_write = CLASSIFIER_ID + "/control/write"  # Write only for client

subscribed_topics = [(topic_control_read, 1)]

client = mqtt.Client(DEVICE_ID)

######################################################################################################################################
########################################################## Control Handlers ##########################################################
######################################################################################################################################

def handler_nfr(data):
    if data is None or len(data) == 0:
        print("This Person is a stranger!")
    else:
        print("This Person is not allowed here, Penalty!")

def handler_nfq(data):
    data = pickle.dumps(data)
    client.publish(topic_control_write, data)
    print("Data published on !" + topic_control_write)
    
def handler_update_database(data):
    # We recieve database dump as pickle file
    # TODO: Update databse
    print("Updated Database")

def handler_update_model(data):
    # We recieve model dump as pickle file
    # TODO: Update Model
    f = open(model_location, 'wb')
    f.write(model_location)
    f.close()
    print("Updated Model")

######################################################################################################################################
########################################################### MQTT Callbacks ###########################################################
######################################################################################################################################

# Message payloads would be in the following format :
#       <msg call id in first three bytes>:<data>
control_descriptors_table = {'read':{
                                        b'nfr': handler_nfr,
                                        b'upd': handler_update_database,
                                        b'upr': handler_update_model
                                    },
                            'write':{ 
                                        b'nfq': handler_nfq
                                    }}

#       nfq -> Not Found Query: Sent by this client to the server along with vector in the case when the vector isn't in the DB
#       nfr -> Not Found Results: Sent by the server along with data in response to 'nfw' request. data is null if vector is not in
#               the database

#       upd -> Update Database: Update the database
#       upm -> Update Model: Update the model


def control_write(callid, data):
    control_descriptors_table['write'][callid](data)
    print("Written "+str(callid))

def control_read(callid, data):
    control_descriptors_table['read'][callid](data)
    print("Recieved "+str(callid))

######################################################################################################################################
########################################################## MQTT Connection ###########################################################
######################################################################################################################################

topic_descriptor_table = {
                            topic_control_read : control_read
                        }

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(subscribed_topics)

# The callback for when a PUBLISH message is received from the server.

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    callid = msg.payload[:3]
    data = msg.payload[3:]
    topic_descriptor_table[msg.topic](callid, data)
    print("Message serviced...")

client.on_connect = on_connect
client.on_message = on_message



######################################################################################################################################
######################################################### Main Program Loop ##########################################################
######################################################################################################################################



client.connect(args.broker.strip(), int(args.port))
print("Connecting to MQTT Broker")
client.subscribe(subscribed_topics)
client.loop_start()

print("System initialization completed... Launching main loop")

for i in processedStream():
    frame, vec, detect, name = i 
    if detect is not False and name != 'Stranger':
        uvecs = db.getUserVectors(name)
        val = validateSimilarity(uvecs, vec)
        if val is True:
            text = name
            client.publish(topic_attendence, name.encode('utf-8'))
        elif val is False:
            # The Person is not recognized. Need to send the data to server
            text = "Stranger"
            print("Person not recognized")
            cv2.putText(frame, "(maybe " + name + ")", (10, 340), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 4, cv2.LINE_AA)
            client.publish(topic_control_write, b'nfq' + pickle.dumps(vec))
            # TODO: Wait for the server's response, Remove this to make it asynchronous
        cv2.putText(frame, text, (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 4, cv2.LINE_AA)
    else:
        print("No Face detected, Maybe come closer!")
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

client.disconnect()
client.loop_stop()
print("Oops, seems the program reached the last!")