import os

#生成train.txt
xml_dir  = 'D:/github/Online_Tournament/Car2024/annotations'
img_dir = 'D:/github/Online_Tournament/Car2024/images'
path_list = list()

# img 文件名
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img)
    xml_path = os.path.join(xml_dir,img.replace('jpg', 'xml'))
    path_list.append((img_path, xml_path))

train_f = open('D:/github/Online_Tournament/Car2024/train.txt', 'w')
val_f = open('D:/github/Online_Tournament/Car2024/val.txt', 'w')

for i ,content in enumerate(path_list):
    img, xml = content
    text = img + ' ' + xml + '\n'
    if i % 5 == 0:
        val_f.write(text)
    else:
        train_f.write(text)
train_f.close()
val_f.close()
