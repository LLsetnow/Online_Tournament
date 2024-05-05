import os

image_path = "D:\github\Online_Tournament\my_dataset\\tumble\\images"
dataset_path = "D:\github\Online_Tournament\my_dataset\\tumble"
txt_path = os.path.join(dataset_path, 'all.txt')
files = os.listdir(image_path)

with open(txt_path, "w") as f:
    for file in files:
        if(file.endswith(".jpg")):
            add_path = os.path.join(image_path, file)
            f.write(f"{add_path}\n")
    f.close()
print(f"已将{image_path}内图像路径写入{txt_path}")


