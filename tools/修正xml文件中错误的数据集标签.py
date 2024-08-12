import os
import xml.etree.ElementTree as ET


def replace_object_name_in_xml(folder_path, original_label, change_label):
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name')
                if name is not None and name.text == original_label:
                    name.text = change_label

            tree.write(file_path)
            print(f"Processed {filename}")


# 替换为你的文件夹路径
folder_path = 'C:\\Users\\admin\Desktop\prop1\prop1\prop1'

# 原标签
original_label = 'prop1'

# 更改后标签
change_label = 'prop'
replace_object_name_in_xml(folder_path, original_label, change_label)
