from lib_machine import fish_detection 
from lib_machine import Type_fish_detection
from lib_machine import send_line
import csv

x_img , y_img , num = fish_detection()

type_fish = Type_fish_detection()

with open('autodetect_data.csv', 'w',newline='') as csvFile:
                fieldnames = ['type_fish','width','height','num']
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({'type_fish':type_fish,'width':x_img,'height':y_img,'num':num})
            
