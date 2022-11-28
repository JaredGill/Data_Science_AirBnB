from genericpath import exists
from PIL import Image

import boto3
import cv2
import os


def resize_images(raw_image_folder_path: str = "C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/unstructured", 
                    processed_image_folder_path: str = "C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images"):
    '''
    Function checkes if image is RGB then resizes height and width scaled from the smallest image in raw_image_folder_path.
    It creates a parent directory for new images to be saved to. 
    Then visits raw folder to retrieve the files and their paths (#The first file in lists is 002717f3-3d2b-44f6-aa98-bc904670883e-a.png).
    The paths are iterated through to find the smallest height using cv2.imread(). 
    Subsequently the paths and files are both iterated through with zip() to resize the image and save to the new directory.
    '''
    img_files = []
    img_paths = []
    img_heights = []
    img_widths = []
    if not os.path.exists(processed_image_folder_path):
            os.makedirs(processed_image_folder_path)
    
    for root, dirs, files in os.walk(raw_image_folder_path):
            img_files.extend(files)

            for file in files:
                img_paths.append(os.path.join(root, file))
                
    
    for path in img_paths:
        image = cv2.imread(path)
        height, width = image.shape[:2]
        img_heights.append(height)
        img_widths.append(width)
    
    smallest_height = min(img_heights)
    print(smallest_height)
    for (file, path) in zip(img_files, img_paths):
        image = cv2.imread(path)
        img = Image.fromarray(image)
        if img.mode == "RGB":
            height, width = image.shape[:2]
            scale_percent = smallest_height/height*100
            new_height = int(image.shape[0]*scale_percent/100)
            new_width = int(image.shape[1]*scale_percent/100)
            dsize = (new_width, new_height)
            output = cv2.resize(image, dsize)
            if not os.path.exists(f'C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images/{file}'):
                cv2.imwrite(f'C:/Users/jared/AiCore/DS_Airbnb/AirbnbDataSci/processed_images/{file}', output)
        else:
            continue
    
if __name__ == "__main__":
    resize_images()