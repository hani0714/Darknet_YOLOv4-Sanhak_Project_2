import os
file_path = 'txt' #파일 생성할 경로
file_names = os.listdir(file_path)
file_names

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(format(i, '04')) + '.txt' #'.jpg' 등 원하는 확장자
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
