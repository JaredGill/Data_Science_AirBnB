from genericpath import exists
from PIL import Image

import boto3
import cv2
import os

#def download_images():

# runs through every iamge height and returns smallest as variable to use to resize and save images to new folder.
def resize_images():
    raw_image_folder_path = f"C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/unstructured"
    processed_image_folder_path = f"C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images"
    img_files = []
    img_paths = []
    img_heights = []
    img_widths = []
    if not os.path.exists(processed_image_folder_path):
            os.makedirs(processed_image_folder_path)
    
    for root, dirs, files in os.walk(raw_image_folder_path):
            #Use .extend here as .append gives a list of lists [[]]
            #File names
            img_files.extend(files)

            # The second loop saves the complete path for each file to a list.
            for file in files:
                #Full path of files e.g. C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/unstructured\\fff65af2-8386-4551-bbe8-4bdead026533\\fff65af2-8386-4551-bbe8-4bdead026533-e.png'
                img_paths.append(os.path.join(root, file))
                
    
    # print(img_files)
    # print(img_paths)
    # loop to discover the smallest height from all images
    for path in img_paths:
        #cv2.imread loads image from specific file path
        image = cv2.imread(path)
        height, width = image.shape[:2]
        img_heights.append(height)
        img_widths.append(width)

    # print(img_files[0])
    # print(img_paths[0])
    # print(img_heights[0])
    # print(img_widths[0])
    # print(len(img_files))
    # print(len(img_paths))
    # print(len(img_heights))
    #The first file in lists is 002717f3-3d2b-44f6-aa98-bc904670883e-a.png
    #Length of all 3 lists is 4700 for the 4700 files in the directory "unstructured"
    
    smallest_height = min(img_heights)
    print(smallest_height)
    for (file, path) in zip(img_files, img_paths):
        image = cv2.imread(path)
        img = Image.fromarray(image)
        # print(img.mode)
        if img.mode == "RGB":
            height, width = image.shape[:2]
            scale_percent = smallest_height/height*100
            new_height = int(image.shape[0]*scale_percent/100)
            new_width = int(image.shape[1]*scale_percent/100)
            # print(f"new height = {new_height}")
            # print(f"new width = {new_width}")
            dsize = (new_width, new_height)
            output = cv2.resize(image, dsize)
            if not os.path.exists(f'C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images/{file}'):
                cv2.imwrite(f'C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images/{file}', output)
        else:
            continue
    
if __name__ == "__main__":
    resize_images()