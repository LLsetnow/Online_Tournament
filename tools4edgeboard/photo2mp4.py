import cv2
import os
import glob
import re

def extract_number(filename):
    # Use regular expression to extract numbers from filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def images_to_video(input_folder, output_file, fps=30):
    # Get all image files in the input folder
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    
    # Sort the image files by the number in the filename
    image_files.sort(key=extract_number)

    # Check if there are any image files
    if not image_files:
        print("No image files found.")
        return

    # Read the first image to get the frame dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print("Unable to read the image file:", image_files[0])
        return

    height, width, layers = first_image.shape

    # Define the video codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print("Unable to read the image file:", image_file, "skipping...")
            continue
        video.write(img)

    # Release the video writer
    video.release()
    print("Video has been saved to", output_file)

# Modify these paths to your desired input folder and output file
input_folder = '../photo/yuan_photo'
output_file = '../res/samples/outputVideo1.mp4'

# Call the function to convert images to video
images_to_video(input_folder, output_file)
