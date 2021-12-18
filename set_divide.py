import os
import math
import random

train=80
valid=20

file_path = 'darknet\\img\\'

file_path_as_originSlash = ""
for  i in range(len(file_path)) :
    if file_path[i] == "\\" :
        file_path_as_originSlash = file_path_as_originSlash + "/"
    else :
        file_path_as_originSlash = file_path_as_originSlash + file_path[i]

file_names = os.listdir(file_path)
file_len = len(file_names)
i=0
while i<file_len:
    temp_fileName = file_names[i]
    if temp_fileName[-4:] == ".txt":
        file_names.pop(i)
        file_len = len(file_names)
        #print(str(i) + "   " + str(file_len) + "    " + temp_fileName)
    else: i += 1

group=[]
for i in range(0, math.ceil(file_len)):
    group.append('/home/' + file_path_as_originSlash + file_names[i])
random.shuffle(group)

print("****************** 1. Train Set 목록(비율: " + str(train) + "%) ******************")

for i in range(0, math.ceil(len(group) * (train/100))) :
    print(group[i])

final_index = math.ceil(len(group) * (train/100))

print("****************** 2. Validation Set 목록(비율: " + str(validation) + "%) ******************")

for i in range(final_index, len(group)) :
    print(group[i])
