import os
import json
from pathlib import Path
import sys

IMAGES_PATH = r'D:\Usuario\Desktop\Base_de_dados\CATCH\IMAGES'
ANNOTATIONS_PATH = r'D:\Usuario\Desktop\Base_de_dados\CATCH\ANNOTATION'
JSON_ANNOTATION = r'D:\Usuario\Desktop\Base_de_dados\CATCH\JSON_ANNOTATION'

images = os.listdir(IMAGES_PATH)

with open(os.path.join(JSON_ANNOTATION, 'CATCH.json'), 'r') as file:
    data = json.load(file)

images_ids = []
images_names = []

for image in data['images']:
    if image['file_name'] in images:
        images_names.append(image['file_name'])
        images_ids.append(int(image['id']))

# instead of a list, use a dict keyed by image_id (much easier!)
polygons_by_image = {}

for annotation in data['annotations']:
    img_id = int(annotation['image_id'])
    if img_id not in images_ids:
        continue  # skip annotations for images we don't have

    # if we don't have this image yet, create the structure
    if img_id not in polygons_by_image:
        index_id = images_ids.index(img_id)
        polygons_by_image[img_id] = {
            "image_id": img_id,
            "file_name": images_names[index_id],
            "cancer_polygons": [],
            "not_cancer_polygons": [],
        }

    polygon_annotation = polygons_by_image[img_id]

    # classify category
    cat_id = int(annotation['category_id'])
    if cat_id in [1, 2, 3, 4, 5, 6]:
        polygon_annotation["not_cancer_polygons"].append(annotation["segmentation"])
    elif cat_id in [7, 8, 9, 10, 11, 12, 13]:
        polygon_annotation["cancer_polygons"].append(annotation["segmentation"])
    else:
        print(f'Annotation is invalid - ANNOTATION : {annotation}')

# turn dict -> list if you still want a list
list_polygons = list(polygons_by_image.values())

# write one JSON per image
for item_json in list_polygons:
    json_path = str(item_json['file_name']).replace('.svs', '.json')
    json_path = os.path.join(ANNOTATIONS_PATH, json_path)
    with open(json_path, "w") as json_file:
        json.dump(item_json, json_file)
        print(f'File {json_path} created!')
