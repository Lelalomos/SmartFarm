#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from lib_machine import send_line
from lib_machine import fish_detection
from lib_machine import sensor_data
from lib_machine import predict_hour
from lib_machine import predict_food
from lib_machine import predict_minute
from lib_machine import predict_food_all_fish
from lib_machine import predict_hour_allfish
from lib_machine import predict_minute_allfish
from lib_machine import feed_food2fish
from lib_machine import feed_machine
from lib_machine import cleanAndExit
from lib_machine import check_loss_food
from lib_machine import food_detection
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
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)


def Process():
    print('เครื่องทำงาน')
    type_fish_in_system = [1,2,3]
    check_pre_HM = True
    hour = 0
    minute = 0
    get_widthfish = 0
    get_heightfish = 0
    get_numfish = 0
    get_typefish = 0
    dir_path = os.getcwd()

    # get token line
    token_line = pd.read_csv('setting_page.csv')
    token_line_data = token_line.Token
    print('line:',token_line_data[0])

    # send_line('ระบบกำลังตรวจสอบไฟล์บันทึกชนิดปลา',token_line_data[0])
    for filename in os.listdir(dir_path):
        filenames = filename.split('.',1)
        if filenames[0] == 'setting_feedfish':
            check_file_setting = True

    if check_file_setting == True:
        # pull data from csv for type fish 
        data_typeCSV = pd.read_csv('setting_feedfish.csv')
        get_typefish = data_typeCSV.type_fish[0]

    while True:

        if check_pre_HM == True:
            # send_line('ระบบกำลังตรวจจับปลา',token_line_data[0])
            get_widthfish , get_heightfish , get_numfish = fish_detection()

            # send_line('ระบบกำลังตรวจจับสภาพแวดล้อม',token_line_data[0])
            light , ph , temp = sensor_data() #sensor data
                
            # send_line('ระบบกำลังทำนายปริมาณอาหาร',token_line_data[0])
            if get_typefish in type_fish_in_system or get_typefish in type_fish_in_system or get_typefish in type_fish_in_system:
                hour = predict_hour(ph,temp,get_heightfish,get_widthfish,light,get_numfish,get_typefish)
                minute = predict_minute(ph,temp,get_heightfish,get_widthfish,light,get_numfish,get_typefish)
            else:
                hour = predict_hour_allfish(ph,temp,get_heightfish,get_widthfish,light,get_numfish)
                minute = predict_minute_allfish(ph,temp,get_heightfish,get_widthfish,light,get_numfish)
            check_pre_HM = False
            send_line('เวลาที่ระบบจะทำการให้อาหารปลาคือ {} โมง : {} นาที'.format(int(hour),int(minute)),token_line_data[0])
            print(int(hour),int(minute))
            
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        split_time = current_time.split(':')
       # s1 = 18
        if int(split_time[0]) == int(hour) and int(split_time[1]) == int(minute):
       # if s1 == 18:

            # take photo
            light , ph , temp = sensor_data() #sensor data
            
            # check model for predict food
            if get_typefish in type_fish_in_system or get_typefish in type_fish_in_system or get_typefish in type_fish_in_system:
                amount_of_food = predict_food(ph,temp,get_heightfish,get_widthfish,light,get_numfish,get_typefish) #predict food result
            else:
                amount_of_food = predict_food_all_fish(ph,temp,get_heightfish,get_widthfish,light,get_numfish)

            send_line('อาหารปลาที่ต้องให้คือปริมาณ: {}'.format(amount_of_food),token_line_data[0])

            # weigh
            EMULATE_HX711=False

            if not EMULATE_HX711:
                import RPi.GPIO as GPIO
                from hx711 import HX711
            else:
                from emulated_hx711 import HX711

            hx = HX711(5, 6)
            hx.set_reading_format("MSB", "MSB")
            hx.set_reference_unit(-2145.590740740741)
            hx.reset()
            hx.tare()
            w=0
            start_test= 0
            end_test=10
            send_line('เครื่องเริ่มทำการให้อาหารปลา',token_line_data[0])
            while w<amount_of_food:
                try:
                    if start_test >= end_test:
                        feed_machine()
                        time.sleep(5)
                        sums = 0
                        for i in range(10):
                            val = hx.get_weight(5)
                            sums += val
                            hx.power_down()
                            hx.power_up()
                            time.sleep(0.1)
                        sums = sums / 10
                        w=sums
                        print(w)
                    else:
                        val = hx.get_weight(5)
                        start_test +=1
                        hx.power_down()
                        hx.power_up()
                        time.sleep(0.1)

                except (KeyboardInterrupt, SystemExit):
                    cleanAndExit()

            # send_line('เครื่องกำลังปล่อยอาหารลงบ่อ',token_line_data[0])
            time.sleep(3)

            # feed step 2
            feed_food2fish()

            warning_text = []
            # check food
            Amount_of_food_left = check_loss_food()
            if Amount_of_food_left <18:
                warning_text.append('feed2machine')
                send_line('ควรเติมอาหารปลา',token_line_data[0])

            if ph < 6.5 or ph > 9:
                warning_text.append('change_water')
                send_line('ควรเปลี่ยนน้ำปลา',token_line_data[0])

            # send_line('กำลังบันทึกข้อมูลการให้อาหาร',token_line_data[0])
           
            time.sleep(900)
            # check food
            food = food_detection()
            if food > 1:
                warning_text.append('food')
                send_line('มีอาหารเหลือ',token_line_data[0])

            if temp < 25:
                warning_text.append('low_temp')
                send_line('น้ำมีอุณหภูมิต่ำกว่ามาตราฐาน',token_line_data[0])
            elif temp >32:
                warning_text.append('height_temp')
                send_line('น้ำมีอุณหภูมิสูงกว่ามาตราฐาน',token_line_data[0])

            # save data feed in csv
            filename_feed = 'feed_data.csv'
            filename_warning = 'warning_data.csv'
            file_exists_feed = os.path.isfile(filename_feed)
            file_exists_warning = os.path.isfile(filename_warning)

#           save feed data 
            with open (filename_feed, 'a',newline='') as csvfile:
                headers = ['hour','minute','ph', 'temp','height','width','light','done','num_fish','type_fish','date']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)

                if not file_exists_feed:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'hour':int(hour),'minute':int(minute),'ph': ph, 'temp': temp,'height':get_heightfish,'width':get_widthfish,'light':light,'done':w,'num_fish':get_numfish,'type_fish':get_typefish,'date': now.strftime("%d/%m/%Y")})

#           save warning data
            with open (filename_warning, 'a',newline='',encoding="UTF-8") as csvfile:
                headers_w = ['date','warning_data']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers_w)

                if not file_exists_warning:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'date': now.strftime("%d/%m/%Y"),'warning_data':warning_text})

            # insert to localhost
            
            check_pre_HM = True
        
        time.sleep(60)

if __name__ == "__main__":
    Process()

