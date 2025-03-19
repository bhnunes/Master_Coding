import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime
from multiprocessing import Pool
import random
import argparse
from dotenv import load_dotenv
import json
import warnings
import cv2
import numpy as np


load_dotenv(override=True)

# Constants from environment variables
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE'))
STRIDE = int(os.getenv('STRIDE'))
MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE'))
TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE'))
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        #import openslide
        from openslide import OpenSlide
else:
    #import openslide
    from openslide import OpenSlide

# Pre-calculate kernel for tissue percentage calculation
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def updateMax(Xmax, Xmin, Ymax, Ymin, x, y):
    Xmax = max(Xmax, x)
    Xmin = min(Xmin, x)
    Ymax = max(Ymax, y)
    Ymin = min(Ymin, y)
    return Xmax, Xmin, Ymax, Ymin


def calculateTissuePercentage(window_np):
    img_bgr = cv2.cvtColor(window_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue_area = sum(cv2.contourArea(contour) for contour in contours)
    tile_area = window_np.shape[0] * window_np.shape[1]
    return (tissue_area / tile_area) >= TISSUE_PERCENTAGE


def generate_window_coordinates(image_width, image_height, window_size, stride, Xmax, Xmin, Ymax, Ymin):
    x_indices = np.arange(0, image_width - window_size + 1, stride)
    y_indices = np.arange(0, image_height - window_size + 1, stride)

    valid_x = (x_indices + window_size > Xmin) & (x_indices < Xmax)
    valid_y = (y_indices + window_size > Ymin) & (y_indices < Ymax)

    valid_x_indices = x_indices[valid_x]
    valid_y_indices = y_indices[valid_y]

    windows = np.array(np.meshgrid(valid_x_indices, valid_y_indices)).T.reshape(-1, 2)
    windows = np.hstack((windows, windows + window_size))

    return windows.tolist()


def process_window(args):
    path_Image, path_cancer_folder, path_not_cancer_folder, patient, x, y, window_size, annotations_cancer, annotations_not_cancer, MATCH_PERCENTAGE = args

    try:
        slide = OpenSlide(path_Image)
        window = slide.read_region((x, y), 0, (window_size, window_size)).convert("RGB")
        window_np = np.array(window)
        slide.close()

        for annotation, label in [(Polygon(annotation), 'cancer') for annotation in annotations_cancer] + [(Polygon(annotation), 'not_cancer') for annotation in annotations_not_cancer]:
            intersection = annotation.intersection(Polygon([(x, y), (x + window_size, y), (x + window_size, y + window_size), (x, y + window_size)]))
            if float((intersection.area / (window_size * window_size))) >= MATCH_PERCENTAGE:
                if calculateTissuePercentage(window_np):
                    patch_image = Image.fromarray(window_np)
                    now = datetime.now()
                    timestamp = now.strftime('%m_%d_%Y_%H_%M_%S')
                    random_number = random.randint(1, 10000)
                    file_path = f"{label.upper()}_PATIENT_{patient}_{random_number}_{timestamp}.png"
                    folder = path_cancer_folder if label == 'cancer' else path_not_cancer_folder
                    patch_image.save(os.path.join(folder, file_path))
                    return True, None

        return True, None
    except Exception as e:
        return False, str(e)
    

def createWindows(path_Image, cancer_color, not_cancer_color):
    Xmax, Xmin, Ymax, Ymin = 0, float('inf'), 0, float('inf')

    path_Annotation = path_Image.replace('.svs', '.xml', 1)
    slide = OpenSlide(path_Image)
    image_width, image_height = slide.dimensions
    slide.close()

    annotations_tree = ET.parse(path_Annotation)
    annotations_root = annotations_tree.getroot()

    annotations_cancer = []
    annotations_not_cancer = []

    for annotation in annotations_root.findall('Annotation'):
        if annotation.get('LineColor') == str(cancer_color):
            layer_region = annotation.findall("Regions//Region")
            for region in layer_region:
                temp = []
                vertices = region.findall("Vertices//Vertex")
                for vertex in vertices:
                    x = float(vertex.get("X"))
                    y = float(vertex.get("Y"))
                    Xmax, Xmin, Ymax, Ymin = updateMax(Xmax, Xmin, Ymax, Ymin, x, y)
                    temp.append((x, y))
                annotations_cancer.append(temp)
        elif annotation.get('LineColor') == str(not_cancer_color):
            layer_region = annotation.findall("Regions//Region")
            for region in layer_region:
                temp = []
                vertices = region.findall("Vertices//Vertex")
                for vertex in vertices:
                    x = float(vertex.get("X"))
                    y = float(vertex.get("Y"))
                    Xmax, Xmin, Ymax, Ymin = updateMax(Xmax, Xmin, Ymax, Ymin, x, y)
                    temp.append((x, y))
                annotations_not_cancer.append(temp)

    windows = generate_window_coordinates(image_width, image_height, WINDOW_SIZE, STRIDE, Xmax, Xmin, Ymax, Ymin)
    return windows, annotations_not_cancer, annotations_cancer
     
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    try:
        parser = argparse.ArgumentParser(description='Process images')
        parser.add_argument('--path_Image', type=str, help='Path to the image file')
        parser.add_argument('--path_cancer_folder', type=str, help='Path to the cancer folder')
        parser.add_argument('--path_not_cancer_folder', type=str, help='Path to the not cancer folder')
        parser.add_argument('--cancer_color', type=str, help='Color code for cancer')
        parser.add_argument('--not_cancer_color', type=str, help='Color code for not cancer')
        parser.add_argument('--patient', type=str, help='Patient ID')
        args = parser.parse_args()

        path_Image = args.path_Image
        path_cancer_folder = args.path_cancer_folder
        path_not_cancer_folder = args.path_not_cancer_folder
        cancer_color = args.cancer_color
        not_cancer_color = args.not_cancer_color
        patient = args.patient

        windows, annotations_not_cancer, annotations_cancer = createWindows(path_Image, cancer_color, not_cancer_color)

        args = [(path_Image, path_cancer_folder, path_not_cancer_folder, patient, x, y, WINDOW_SIZE, annotations_cancer, annotations_not_cancer, MATCH_PERCENTAGE) for x, y, _, _ in windows]

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(process_window, args)

        for success, message in results:
            if not success:
                raise Exception(message)

        status = 'COMPLETED'
        comments = ''
    except Exception as e:
        status = 'FAILED'
        comments = str(e)
    finally:
        data = {
            "status": status,
            "comments": comments
        }
        json_data = json.dumps(data)
        print(json_data)