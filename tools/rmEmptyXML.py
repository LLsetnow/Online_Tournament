import os
import xml.etree.ElementTree as ET


def delete_empty_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filepath = os.path.join(directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Check if there is an object tag in the xml file
            if not root.findall('object'):
                # No object tag found, prepare to delete both xml and jpg files
                os.remove(filepath)
                jpg_filename = filename.replace(".xml", ".jpg")
                jpg_filepath = os.path.join(directory, jpg_filename)

                # Check if the corresponding jpg file exists before deleting
                if os.path.exists(jpg_filepath):
                    os.remove(jpg_filepath)
                    print(f"Deleted: {filepath} and {jpg_filepath}")
                else:
                    print(f"Deleted: {filepath} (No corresponding JPG found)")


folder_path = 'D:\github\Online_Tournament\my_dataset'
dataset = 'good_v4'
directory = os.path.join(folder_path, dataset, 'annotations')
delete_empty_files(directory)
