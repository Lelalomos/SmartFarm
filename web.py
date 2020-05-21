from flask import Flask, render_template, flash, url_for, request, redirect , session
from form import submit_data_fish
from lib_machine import echo_type_fish
import requests
import csv
import os
import sys
import time
import pandas as pd
from lib_machine import send_line
import subprocess
import multiprocessing


app = Flask(__name__)
app.config['SECRET_KEY'] = '65a7ae36981109ed2d9f59024b687de9'  # Encypt
fish_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = fish_folder
check_machine_file_run = False


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
        },
        {
            'name': 'ปลาการ์ตูน',
            'id': 4
        },
        {
            'name': 'ปลาหมอสี',
            'id': 5
        },
        {
            'name': 'ปลามังกร หรือ ปลาอะโรวาน่า',
            'id': 6
        },
        {
            'name': 'ปลาเทวดา',
            'id': 7
        },
        {
            'name': 'ปลาสิงโตปีก',
            'id': 8
        },
        {
            'name': 'ปลาสอดหางดาบ',
            'id': 9
        },
        {
            'name': 'ปลาเสือเยอรมัน',
            'id': 10
        },
        {
            'name': 'ปลาม้าลาย',
            'id': 11
        },
        {
            'name': 'ปลาทั่วไป',
            'id': 12
        }
    ]


@app.route("/index", methods=['GET', 'POST'])
def index():
    check_file_auto = False
    check_file_setting_page = False
    dir_path = os.getcwd()

# check setting line
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_page':
            check_file_setting_page = True
    
    if check_file_setting_page == False:
        return redirect(url_for('setting_page'))

    # read token line
    if check_file_setting_page == True:
        token_line = pd.read_csv('setting_page.csv')
        token_line_data = token_line.Token
        session['Token_line'] = token_line_data[0]
    
    send_line('ระบบกำลังงาน 192.168.1.99:6677',session['Token_line'])

    # check autodetect file
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'autodetect_data':
            check_file_auto = True
    
    #check setting_feed file
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_feedfish':
            return redirect(url_for('machine_work'))
            break
    
    if check_file_auto == False:
        os.system(
        "sudo python3 /home/pi/tf/tensorflow1/models/research/object_detection/autodetect.py")

    flash('ระบบตรวจสอบอัตโนมัติเสร็จสิ้น','success')
    data_typeCSV = pd.read_csv('autodetect_data.csv')
    id_fish = data_typeCSV.type_fish[0]
    num = data_typeCSV.num[0]
    x_img = data_typeCSV.width[0]
    y_img = data_typeCSV.height[0]

    # take photo img
    full_filename_fishimg = os.path.join(app.config['UPLOAD_FOLDER'], 'image_fish.png')
    full_filename_machine = os.path.join(app.config['UPLOAD_FOLDER'], 'machine.gif')

    fish_id , fish_name = echo_type_fish(id_fish)

    return render_template('index.html', fish=fish,num = num,id_fish=fish_id,x_img=x_img,y_img=y_img,name_fish = fish_name,image_fish = full_filename_fishimg,image_machine = full_filename_machine)


@app.route("/machine_work",methods = ['GET','POST'])
def machine_work():
    check_setting_file = False #check setting_file not exits
    check_feeddata_file = False #check feed_data file not exits
    check_btn_search = False
    check_file_setting_page = False
    check_error = True
    check_warning_data = False 
    list_warning = []
    list_war_text = []
    _len_warning =0 

    dir_path = os.getcwd()
    if request.method == 'POST':
        if request.form['submit_fishdetection'] == 'submit':
            option_selected = request.form.get('fish')
            num_fish = request.form.get('num')
            width_fish = request.form.get('width')
            height_fish = request.form.get('height')
            type_selection = option_selected
            
            # create csv 
            with open('setting_feedfish.csv', 'w',newline='') as csvFile:
                fieldnames = ['type_fish']
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'type_fish':type_selection})
            check_setting_file = True

            fish_id , fish_name = echo_type_fish(type_selection)
            send_line('ได้ทำการยืนยันข้อมูลปลา ดังนี้\n ชนิดปลา: {} \n ความยาวของปลา: {} \n ความสูงของปลา: {} \n จำนวนปลา: {} ตัว \n '.format(fish_name,width_fish,height_fish,num_fish),session['Token_line'])


# check setting line
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_page':
            check_file_setting_page = True
    
    if check_file_setting_page == False:
        return redirect(url_for('setting_page'))

    # read token line
    if check_file_setting_page == True:
        token_line = pd.read_csv('setting_page.csv')
        token_line_data = token_line.Token
        session['Token_line'] = token_line_data[0]


# find setting file
    if session.get('check_start_setting') != True:
        for filename in os.listdir(dir_path):
            filenames = filename.split('.', 1)
            if filenames[0] == 'setting_feedfish':
                check_setting_file = True

# fine feed_data file
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'feed_data':
            check_feeddata_file = True

# check warning data
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'warning_data':
            check_warning_data = True

    if check_warning_data == True:
        
        read_warning_data = pd.read_csv('warning_data.csv')
        _warning_data = read_warning_data.warning_data
        _splt = _warning_data[0].split(",")
        _len_warning = len(_splt)
        for i in range(_len_warning-1):
            _split_data_warning =  _splt[i][2:-1]
            list_warning.append(_split_data_warning)

        for list_war in list_warning:
            if 'feed2machine' == list_war:
                list_war_text.append('ควรเติมอาหารปลา')
            elif 'change_water' == list_war:
                list_war_text.append('ควรเปลี่ยนน้ำปลา')
            elif 'food' == list_war:
                list_war_text.append('มีอาหารเหลือ')
            elif 'low_temp' == list_war:
                list_war_text.append('น้ำมีอุณหภูมิต่ำกว่ามาตราฐาน')
            elif 'height_temp' == list_war:
                list_war_text.append('น้ำมีอุณหภูมิสูงกว่ามาตราฐาน')
        # print(list_war_text)
    # check setting file
    if check_setting_file == True or session.get('check_start_setting') == True :
        # check refresh page
        # check warning data
        check_machine_process = False #not create file
        for filename in os.listdir(dir_path):
            filenames = filename.split('.', 1)
            if filenames[0] == 'machine_process':
                check_machine_process = True


        if check_machine_process == False :
            with open('machine_process.csv', 'w',newline='') as csvFile:
                    fieldnames = ['process']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'process':'1'})
            subprocess.Popen(['sudo python3 /home/pi/tf/tensorflow1/models/research/object_detection/machine.py'],shell = True)
            
           

        # read csv to show graph 
        tfish = pd.read_csv('setting_feedfish.csv')
        data_type = tfish.type_fish[0]

        if check_feeddata_file == True :
            data_graph = pd.read_csv('feed_data.csv')

            data_graph_SeTypeFish = data_graph[data_graph.type_fish == data_type]
            ph_graph = data_graph_SeTypeFish.ph
            temp_graph = data_graph_SeTypeFish.temp
            light_graph = data_graph_SeTypeFish.light
            done_graph = data_graph_SeTypeFish.done
            date_graph = data_graph_SeTypeFish.date

            # convert to list
            ph_graph = ph_graph.values.tolist()
            temp_graph = temp_graph.values.tolist()
            light_graph = light_graph.values.tolist()
            done_graph = done_graph.values.tolist()
            date_graph = date_graph.values.tolist()

        else:
            ph_graph = 0
            temp_graph = 0
            light_graph = 0
            done_graph = 0
            date_graph = 0

    else:
        send_line('ระบบหาไฟล์ setting_feedfish ไม่พบ จึงต้องให้ผู้ใช้ทำการกรอกข้อมูลของท่านใหม่อีกครั้ง ขออภัยในความไม่สะดวก ครับ/ค่ะ',session['Token_line'])
        return redirect(url_for('index'))
    full_filename_red = os.path.join(app.config['UPLOAD_FOLDER'], 'red.gif')
    full_filename_green = os.path.join(app.config['UPLOAD_FOLDER'], 'green.gif')
    full_filename_machine_talk = os.path.join(app.config['UPLOAD_FOLDER'], 'client-1.jpg')
    full_filename_left = os.path.join(app.config['UPLOAD_FOLDER'], 'quote_sign_left.png')
    full_filename_right = os.path.join(app.config['UPLOAD_FOLDER'], 'quote_sign_right.png')
    return render_template('machine_work.html', title='machine_work' , ph = ph_graph ,temp = temp_graph ,light = light_graph , done = done_graph , date = date_graph , check_error = check_error , red =full_filename_red , green = full_filename_green , machine_talk = full_filename_machine_talk , list_war = list_war_text , len_list_war = _len_warning , right = full_filename_right , left = full_filename_left )


@app.route("/setting_page",methods = ['GET','POST'])
def setting_page():
    dir_path = os.getcwd()

    # take photo img
    full_filename_classify = os.path.join(app.config['UPLOAD_FOLDER'], 'clf.gif')

    # line img
    full_filename_TokenLine1 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine1.png')
    full_filename_TokenLine2 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine2.png')
    full_filename_TokenLine3 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine3.png')
    full_filename_TokenLine4 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine4.png')
    full_filename_TokenLine5 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine5.png')
    full_filename_TokenLine6 = os.path.join(app.config['UPLOAD_FOLDER'], 'TokenLine6.png')

    if request.method == 'POST':
        if request.form.get('submit_line',False) == 'submit':
            token_line = request.form.get('line_txt')
            print(token_line)
            if token_line == '' or token_line is None:
                with open('setting_page.csv', 'w',newline='') as csvFile:
                    fieldnames = ['Token']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'Token':'empty'})
                session['Token_line'] = 'empty'
                return redirect(url_for('index'))
            else:
                with open('setting_page.csv', 'w',newline='') as csvFile:
                    fieldnames = ['Token']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'Token':token_line})
                session['Token_line'] = token_line
                return redirect(url_for('index'))
    return render_template('setting_page.html' , clf = full_filename_classify , token_line1 =full_filename_TokenLine1,token_line2 = full_filename_TokenLine2,token_line3 =full_filename_TokenLine3,token_line4 = full_filename_TokenLine4,token_line5 = full_filename_TokenLine5,token_line6 = full_filename_TokenLine6  )

@app.route("/",methods = ['GET','POST'])
def welcome():
    dir_path = os.getcwd()
    check_line = False
    
    #check line
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_page':
            check_line = True
    
    # take photo img
    full_filename_start_logo = os.path.join(app.config['UPLOAD_FOLDER'], 'fish_logo.gif')
    return render_template('welcome.html',start_logo = full_filename_start_logo , check_setting = check_line)


if __name__ == "__main__":
    dir_path = os.getcwd()
    check_start_comit = False
    check_start_setting = False

    # find setting file
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_feedfish':
            check_start_comit = True
    
    # check setting line
    for filename in os.listdir(dir_path):
        filenames = filename.split('.', 1)
        if filenames[0] == 'setting_page':
            check_start_setting = True
    
    if check_start_comit == True and check_start_setting == True:
        print('machine_work')
        with open('machine_process.csv', 'w',newline='') as csvFile:
                    fieldnames = ['process']
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({'process':'1'})
        subprocess.Popen(['sudo python3 /home/pi/tf/tensorflow1/models/research/object_detection/machine.py'],shell = True)
        

        app.run(host='192.168.137.113',port=6677,debug=True,threaded=True)
    else:
        print('not work')
        app.run(host='192.168.137.113',port=6677,debug=True,threaded=True)

