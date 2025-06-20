import zmq
import numpy as np
import cv2
import time
import json
from deepface import DeepFace
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def detect(net, img, confidence_threshold):
    #detecting objects
    blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
        
    net.setInput(blob)
    outs = net.forward(outputlayers)

    # get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
            #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)        
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    return {'indexes':indexes,'boxes':boxes, 'class_ids': class_ids}

def draw(img,res):
    boxes = res['boxes']
    indexes = res['indexes']
    class_ids = res['class_ids']
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label,(x,y+30),font,3,(255,255,0),2)

    return img
 
def formatresult(res,width,height):
    boxes = res['boxes']
    indexes = res['indexes']
    class_ids = res['class_ids']
    output = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            x /= width
            w /= width
            y /= height
            h /= height
            output.append({'item':label,'bbox':[x,y,w,h]})
    return output

def objectList(res):
    boxes = res['boxes']
    indexes = res['indexes']
    class_ids = res['class_ids']
    output = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            x /= width
            w /= width
            y /= height
            h /= height
            output.append(label)
    return output

def getObjectSet(list):
    set = {}
    for item in list:
        set[item] = True
    return set

def compareSets(set1, set2):
    out = []
    for item in set2.keys():
        if not item in set1:
            out.append('enter_' + item)
    for item in set1.keys():
        if not item in set2:
            out.append('leave_' + item)
    return out

#Emotion stuff
emotion_img_list = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'suprise': 0, 'neutral': 0}
speech_emotion_list = {'Neutral': 0, 'Negative': 0, 'Positive': 0}

def detect_img_emotion(img, emotion_img_list):
    face_analysis = DeepFace.analyze(img, enforce_detection=False)
    emotion = face_analysis[0]['dominant_emotion']
    if emotion in emotion_img_list:
        emotion_img_list[emotion] = emotion_img_list[emotion] + 1 
    return emotion

def determine_emotion(emotion_img_list, speech_emotion_list):
    positive_emotions = emotion_img_list['happy'] + emotion_img_list['suprise'] + (speech_emotion_list['Positive'] *  100)
    negative_emotions = emotion_img_list['angry'] + emotion_img_list['disgust'] + emotion_img_list['fear'] + emotion_img_list['sad'] + (speech_emotion_list['Negative'] * 10)
    neutral_emotions = emotion_img_list['neutral'] + (speech_emotion_list['Neutral'] * 50)
    first_emotion = max(emotion_img_list, key=emotion_img_list.get)
    secound_emotion = secound_place(emotion_img_list)

    print(f"emotion: {emotion_img_list}")
    print(f"speech: {speech_emotion_list}")
    print(f"first emotion: {first_emotion}")
    print(f"secound emotion: {secound_emotion}")

    if neutral_emotions > negative_emotions and neutral_emotions > positive_emotions:
        return 'neutral'
    elif abs(positive_emotions - negative_emotions) <= 2:
        return 'unknown'
    elif positive_emotions > negative_emotions:
        if abs(emotion_img_list[first_emotion] - emotion_img_list[secound_emotion]) <= 2:
            if secound_emotion in ['happy', 'suprise']:
                return f"{secound_emotion} and {first_emotion}"
        return first_emotion
    elif positive_emotions < negative_emotions:
        if abs(emotion_img_list[first_emotion] - emotion_img_list[secound_emotion]) <= 2:
            if secound_emotion in ['angry', 'disgust', 'fear', 'sad']:
                return f"{secound_emotion} and {first_emotion}"
        return first_emotion

def send_emotion(emotion_img_list,speech_emotion_list):
    emotion = determine_emotion( emotion_img_list, speech_emotion_list)
    emotion = f"emotion_{emotion}"
    outsocket.send_string(emotion)

def determine_speech_emotion(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    speech_emotion_list['Neutral'] = speech_emotion_list['Neutral'] + sentiment_dict['neu']
    speech_emotion_list['Positive'] = speech_emotion_list['Positive'] + sentiment_dict['neu']
    speech_emotion_list['Negative'] = speech_emotion_list['Negative'] + sentiment_dict['neg']

#return the secound highest element in a dictionary
def secound_place(dic):
    first = max(dic, key=dic.get)
    current = min(dic, key=dic.get)
    print(f'min emotion: {current}')
    print(f'max emotion: {first}')
    for element in dic:
        if dic[element] > dic[current] and dic[element] < dic[first]:
            current = element
    return current
#################



# Load configuration
with open('launch.json') as f:
  config = json.load(f)
print(config)

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))


# Setup the sockets
context = zmq.Context()

# Input camera feed from furhat using a SUB socket
insocket = context.socket(zmq.SUB)
insocket.setsockopt_string(zmq.SUBSCRIBE, '')
insocket.connect('tcp://' + config["Furhat_IP"] + ':3000')
insocket.setsockopt(zmq.RCVHWM, 1)
insocket.setsockopt(zmq.CONFLATE, 1)  # Only read the last message to avoid lagging behind the stream.

#input speech socket
Speechsocket = context.socket(zmq.SUB)
Speechsocket.setsockopt_string(zmq.SUBSCRIBE, '')
Speechsocket.connect('tcp://' + config["Furhat_IP"] + ':8888')
Speechsocket.setsockopt(zmq.RCVHWM, 1)
Speechsocket.setsockopt(zmq.CONFLATE, 1)

#set up poller 
poller = zmq.Poller()
poller.register(insocket, zmq.POLLIN)
poller.register(Speechsocket, zmq.POLLIN)

# Output results using a PUB socket
context2 = zmq.Context()
outsocket = context2.socket(zmq.PUB)
outsocket.bind("tcp://" + config["Dev_IP"] + ":" + config["detection_exposure_port"])


prevset = {}
iterations = 0
detection_period = config["detection_period"] # Detecting objects is resource intensive, so we try to avoid detecting objects in every frame
detection_threshold = config["detection_confidence_threshold"] # Detection threshold takes a double between 0.0 and 1.0

print('connected, entering loop')
while True:
    
    socks = dict(poller.poll())

    if insocket in socks:
        string = insocket.recv()
        magicnumber = string[0:3]
        
        # check if we have a JPEG image (starts with ffd8ff)
        if magicnumber == b'\xff\xd8\xff':
            buf = np.frombuffer(string,dtype=np.uint8)
            img = cv2.imdecode(buf,flags=1)

            if (iterations % detection_period == 0):
                buf = np.frombuffer(string,dtype=np.uint8)
                img = cv2.imdecode(buf,flags=1)
                height,width,channels = img.shape
                res = detect(net,img, detection_threshold)
                img2=draw(img,res)
                
                currset = getObjectSet(objectList(res))
                objdiff = compareSets(prevset,currset)
                if len(objdiff):
                    msg = ' '.join(objdiff)
                    outsocket.send_string(msg)

                prevset = currset
                cv2.imshow("yolov3", img2)

            detect_img_emotion(img, emotion_img_list)
            if iterations % 100 == 0:
                send_emotion(emotion_img_list, speech_emotion_list)
                emotion_img_list = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'suprise': 0, 'neutral': 0}
                overall_img_emotion = ()
                speech_emotion_list = {'Neutral': 0, 'Negative': 0, 'Positive': 0}

    if Speechsocket in socks:
        speech = Speechsocket.recv()
        determine_speech_emotion(speech.decode('utf-8').replace('speech_',''))
        
        if speech.decode('utf-8') == 'speech_turn off':
            break
    
    iterations += 1