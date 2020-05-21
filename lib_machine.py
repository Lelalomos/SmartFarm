from PIL import Image
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import time
import sys
import joblib
import numpy as np
import os
from datetime import datetime
import csv

# only RPi
import serial
import RPi.GPIO as GPIO

# line 
import requests

sys.path.append("..")

IM_WIDTH = 1280
IM_HEIGHT = 720
IMAGE_NAME = 'image_fish.png'
CWD_PATH = os.getcwd()
PATH_TO_IMAGE = os.path.join(CWD_PATH,'static','img', IMAGE_NAME)

def take_photo():   #take photo
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,IM_WIDTH)
    ret = cap.set(4,IM_HEIGHT)

    CWD_PATH = os.getcwd() 
    ret, frame = cap.read()
#    time.sleep(20)
    cv2.imwrite('static/img/image_fish.png',frame)
    PATH_2_IMAGE = os.path.join(CWD_PATH,'static','img','image_fish.png')
    img = Image.open(PATH_2_IMAGE)
    img.save(PATH_2_IMAGE, quality=95, dpi=(96, 96))
    cap.release()
    # cv2.destroyAllWindows()


def cleanAndExit():
    if not EMULATE_HX711:
        GPIO.cleanup()
    sys.exit()

def send_line(msg,get_token):
    # read csv file
    url = 'https://notify-api.line.me/api/notify'
    token = get_token
    if token != "empty":
       headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
       r = requests.post(url, headers=headers, data = {'message':msg})

def check_loss_food():
    
    GPIO.setmode(GPIO.BCM)
    TRIG = 23
    ECHO = 24
    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)
    try:
        GPIO.output(TRIG, False)
        time.sleep(2)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        while GPIO.input(ECHO)==0:
            pulse_start = time.time()

        while GPIO.input(ECHO)==1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)
    except (KeyboardInterrupt, SystemExit):
        GPIO.cleanup()

    return distance

def feed_machine():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(14, GPIO.OUT)
    pwm = GPIO.PWM(14, 50)
    pwm.start(5)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(7.5)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(5)
    time.sleep(0.5)


def feed_food2fish():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(2, GPIO.OUT)
    pwm = GPIO.PWM(2, 50)
    pwm.start(7.5)
    time.sleep(3)
    pwm.ChangeDutyCycle(4)
    time.sleep(5)
    pwm.ChangeDutyCycle(7.5)

def fish_detection(): #ทำนายปลา
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    # IMAGE_NAME = 'image_fish.png'

    y_summax = 0
    x_summax = 0

    # object
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME,'fish_detection', 'frozen_inference_graph.pb')
    # Label
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data','fish_detection', 'labelmap.pbtxt')
    # Path to image
    
    take_photo()

    NUM_CLASSES = 1
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(PATH_TO_IMAGE)
    image_dpi = Image.open(PATH_TO_IMAGE)

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, image,
                                                                                             1,
                                                                                             0,
                                                                                             np.squeeze(
                                                                                                 boxes),
                                                                                             np.squeeze(classes).astype(
                                                                                                 np.int32),
                                                                                             np.squeeze(
                                                                                                 scores),
                                                                                             category_index,
                                                                                             use_normalized_coordinates=True,
                                                                                             min_score_thresh=0.3,
                                                                                             line_thickness=8)
    boxs = np.squeeze(boxes)
    score = np.squeeze(scores)
    pre_class = counting_mode.split(':')
    # print(pre_class)
    # pre_class = pre_class.replace("'","")
    font = cv2.FONT_HERSHEY_SIMPLEX
    if(len(counting_mode) == 0):
        cv2.putText(image, "...", (10, 35), font, 0.8,
                (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    else:
        cv2.putText(image, counting_mode, (10, 35), font, 0.8,
                (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)

    height, width, d = image.shape
    for i in range(0, len(score)):
        # print(boxes[[0], [i]])
        if(score[i] >= 0.3):
            ymin, xmin, ymax, xmax = boxs[i]
            x1 = int(xmin*width)
            x2 = int(xmax*width)
            y1 = int(ymin*height)
            y2 = int(ymax*height)

            xmax, ymax = convert_pixel2inch(x2, y2,image_dpi)
            y_summax += ymax
            x_summax += xmax
    pre_class[2] = pre_class[2].strip()
    if int(pre_class[2]) > 1:
        x_avg_summax = x_summax/len(score)
        y_avg_summax = y_summax/len(score)
    print(str(xmax),str(ymax),str(pre_class[2]))
    sess.close() 
    tf.reset_default_graph()
    cv2.destroyAllWindows()
    return xmax,ymax,int(pre_class[2])

def Type_fish_detection(): #ทำนายปลา
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    # object
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME,'type_fish_detection', 'frozen_inference_graph.pb')
    # Label
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data','type_fish_detection', 'labelmap.pbtxt')
    # Path to image
    
    take_photo()
    NUM_CLASSES = 3

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(PATH_TO_IMAGE)
    # image_dpi = Image.open(PATH_TO_IMAGE)

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, image,
                                                                                             1,
                                                                                             0,
                                                                                             np.squeeze(
                                                                                                 boxes),
                                                                                             np.squeeze(classes).astype(
                                                                                                 np.int32),
                                                                                             np.squeeze(
                                                                                                 scores),
                                                                                             category_index,
                                                                                             use_normalized_coordinates=True,
                                                                                             min_score_thresh=0.3,
                                                                                             line_thickness=8)
    pre_class = counting_mode.split(':')
    pre_class[0] = pre_class[0].strip()
    pre_class[0] = pre_class[0][1::]
    sess.close() 
    tf.reset_default_graph()
    cv2.destroyAllWindows()
    return pre_class[0]

def food_detection():
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    # object
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME,'food_detection', 'frozen_inference_graph.pb')
    # Label
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data','food_detection', 'labelmap.pbtxt')
    # Path to image
    
    take_photo()
    NUM_CLASSES = 1

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(PATH_TO_IMAGE)
    # image_dpi = Image.open(PATH_TO_IMAGE)

    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1, image,
                                                                                             1,
                                                                                             0,
                                                                                             np.squeeze(
                                                                                                 boxes),
                                                                                             np.squeeze(classes).astype(
                                                                                                 np.int32),
                                                                                             np.squeeze(
                                                                                                 scores),
                                                                                             category_index,
                                                                                             use_normalized_coordinates=True,
                                                                                             min_score_thresh=0.3,
                                                                                             line_thickness=8)
    pre_food_class = counting_mode.split(':')
    sess.close() 
    tf.reset_default_graph()
    cv2.destroyAllWindows()
    return len(pre_food_class)

def convert_pixel2inch(x2, y2, images):
    img_split = IMAGE_NAME.split('.', 1)
    if img_split[-1] == 'png' or img_split[-1] == 'PNG':
        img = Image.open(PATH_TO_IMAGE)
        dpi = img.info['dpi']
        xmax_inch = round(x2/dpi[0], 2)
        ymax_inch = round(y2/dpi[0], 2)
    else:
        img = Image.open(PATH_TO_IMAGE)
        img_files = img_split[0]+'.png'
        PATH_2_IMAGE = os.path.join(CWD_PATH, img_files)
        img.save(PATH_2_IMAGE, quality=95, dpi=(96, 96))

        img_convert = Image.open(PATH_2_IMAGE)
        dpi = img_convert.info['dpi']
        xmax_inch = round(x2/96, 2)
        ymax_inch = round(y2/96, 2)
    return xmax_inch, ymax_inch


def predict_food(ph,temp,height,width,light,numFish,typeFish):
    data_food_predict = [[ph,temp,height,width,light,numFish,typeFish]]
    df_data = pd.DataFrame(data_food_predict, columns = ['ph','temp','height','width','light','num_fish','type_fish'])
    done = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_Pfish_all.pkl')
    food = done.predict(df_data)
    return food[0]

def predict_food_all_fish(ph,temp,height,width,light,numFish):
    data_food_allF_predict = [[ph,temp,height,width,light,numFish]]
    df_data_allF = pd.DataFrame(data_food_allF_predict, columns = ['ph','temp','height','width','light','num_fish'])
    done_allF = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_done_allFish.pkl')
    food_allF = done_allF.predict(df_data_allF)
    return food_allF[0]

def predict_hour(ph,temp,height,width,light,numFish,typeFish):  #predict fish
    data_hour_predict = [[ph,temp,height,width,light,numFish,typeFish]]
    df_hour = pd.DataFrame(data_hour_predict, columns = ['ph','temp','height','width','light','num_fish','type_fish'])
    model_hour = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_hour_fishInSystem.pkl')
    hour = model_hour.predict(df_hour)
    hour = np.round(hour[0])
    return hour

def predict_minute(ph,temp,height,width,light,numFish,typeFish):
    data_minute_predict = [[ph,temp,height,width,light,numFish,typeFish]]
    df_minute = pd.DataFrame(data_minute_predict, columns = ['ph','temp','height','width','light','num_fish','type_fish'])
    model_minute = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_minute_fishInSystem.pkl')
    minute = model_minute.predict(df_minute)
    minute = np.round(minute[0])
    return minute

def predict_hour_allfish(ph,temp,height,width,light,numFish):
    data_hour_predict_all = [[ph,temp,height,width,light,numFish]]
    df_hour_all = pd.DataFrame(data_hour_predict_all, columns = ['ph','temp','height','width','light','num_fish'])
    model_hour_all = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_hour_fishWithoutSystem.pkl')
    hour_all = model_hour_all.predict(df_hour_all)
    hour_all = np.round(hour_all[0])
    return hour_all

def predict_minute_allfish(ph,temp,height,width,light,numFish):
    data_minute_predict_all = [[ph,temp,height,width,light,numFish]]
    df_minute_all = pd.DataFrame(data_minute_predict_all, columns = ['ph','temp','height','width','light','num_fish'])
    model_minute_all = joblib.load('/home/pi/tf/tensorflow1/models/research/object_detection/model_food/model_minute_fishWithoutSystem.pkl')
    minute_all = model_minute_all.predict(df_minute_all)
    minute_all = np.round(minute_all[0])
    return minute_all

def sensor_data(): 
    _sensorData_temp = []
    _sensorData_light = []
    _sensorData_PH = []

    for i in range(20):
        ser = serial.Serial("/dev/ttyACM0", 9600)
        from_sensor = ser.readline()
        decodeData = from_sensor.decode('utf-8')
        split_dataSensor = decodeData.split(',')
        if len(split_dataSensor) == 3:
            _sensorData_temp.append(split_dataSensor[2].rstrip())
            _sensorData_light.append(split_dataSensor[0])
            _sensorData_PH.append(split_dataSensor[1])

    LIGHT_sensor = []
    TEMP_sensor = []
    PH_sensor = []
    for i in _sensorData_light:
        try:
            i = float(i)
        except ValueError:
            pass
        LIGHT_sensor.append(i)

    for i in _sensorData_temp:
        try:
            i = float(i)
        except ValueError:
            pass
        TEMP_sensor.append(i)
    
    for i in _sensorData_PH:
        try:
            i = float(i)
        except ValueError:
            pass
        PH_sensor.append(i)

    sum_light = sum(LIGHT_sensor[-10:])/10
    sum_ph = sum(PH_sensor[-10:])/10
    sum_temp = sum(TEMP_sensor[-10:])/10

    print(sum_light,sum_ph,sum_temp)
    # light , PH , Temp ; data from arduino to raspberry py
    return sum_light,sum_ph,sum_temp

def echo_type_fish(name_fish):
    fish_id = 0
    fish_name = ''
    fish = [
        {
            'name': 'ปลากัด',
            'id': 1
        },
        {
            'name': 'ปลาทอง',
            'id': 2
        },
        {
            'name': 'ปลาคาร์พ',
            'id': 3
        }]

    if name_fish == 'carp':
        name_fish = 'ปลาคาร์พ'
    elif name_fish == 'betta':
        name_fish = 'ปลากัด'
    else:
        name_fish = 'ปลาทอง'

    for i in range(len(fish)):
        if name_fish == str(fish[i]['name']):
            fish_id = fish[i]['id']
            fish_name = fish[i]['name']
    return fish_id , fish_name

