import pymysql
from dotenv import load_dotenv
import os
import re
import subprocess
import time
import json
from datetime import datetime
import sys

def createfolders():
    os.makedirs(PATH_CANCER_FOLDER, exist_ok=True)
    os.makedirs(PATH_NOT_CANCER_FOLDER, exist_ok=True)
    os.makedirs(PATH_CANCER_MASK_FOLDER, exist_ok=True)
    os.makedirs(PATH_NOT_CANCER_MASK_FOLDER, exist_ok=True)

def open_conn():
    conn = pymysql.connect(
        charset="utf8mb4",
        connect_timeout=30,
        cursorclass=pymysql.cursors.DictCursor,
        db=os.getenv('MYSQL_DB'),
        host=os.getenv('MYSQL_HOST'),
        password=os.getenv('MYSQL_PASSWORD'),
        read_timeout=30,
        port=3306,
        user=os.getenv('MYSQL_USER'),
        write_timeout=30,
    )
    return conn

def getFolder(path):
    try:
        conn=open_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM `image_folder_control` WHERE `STATUS` in ('COMPLETED','TO BE PROCESSED') AND `FILEPATH`=%s",(path))
        data = cursor.fetchall()
        return data
    except Exception as e:
        raise ValueError('Could not execute function getFolder - Error '+str(e))
    finally:
        conn.close()

def createFolderEntry(image, patient):
        try:
            conn=open_conn()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO `image_folder_control`(`FILEPATH`,`PATIENT`,`STATUS`) VALUES (%s,%s,%s)",(image,patient,'TO BE PROCESSED'))
            conn.commit()
        except Exception as e:
            raise ValueError('Could not execute function createFolderEntry - Error '+str(e))
        finally:
            conn.close()

def extract_patient(text):
    pattern = r"\w+_(\d+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return text

def find_svs_files(directory):
    svs_files = []
    patients=[]
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.svs'):
                file_path = os.path.join(root, name)
                svs_files.append(file_path)
                patients.append(extract_patient(root))
    return svs_files, patients

def AddNewCases():
    svs_files, patients=find_svs_files(IMAGESPATH)
    for image, patient in zip(svs_files, patients):
        data=getFolder(image)
        if len(data)==0:
            createFolderEntry(image, patient)

def getCaseToProcess():
    try:
        conn=open_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM `image_folder_control` WHERE `STATUS` = 'TO BE PROCESSED' LIMIT 1")
        data = cursor.fetchall()
        return data
    except Exception as e:
        raise ValueError('Could not execute function getFolder - Error '+str(e))
    finally:
        conn.close()

def getTotalCases():
    try:
        conn=open_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM `image_folder_control` WHERE `STATUS` = 'TO BE PROCESSED'")
        data = cursor.fetchall()
        return data
    except Exception as e:
        raise ValueError('Could not execute function getFolder - Error '+str(e))
    finally:
        conn.close()

def updateCase(id, counter_cancer, counter_not_cancer, execution_time, status, comments, window_size, stride,tissuePercentage, matchPercentage):
    try:
        conn=open_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE `image_folder_control` SET `CANCER_QTD`=%s,`NON_CANCER_QTD`=%s,`PROCESSINGTIME_MINUTES`=%s,`COMMENTS`=%s,`STATUS`=%s,`WINDOW_SIZE`=%s,`STRIDE`=%s, `MATCH_PERCENTAGE`=%s, `TISSUE_PERCENTAGE`=%s  WHERE `ID` = %s",(counter_cancer,counter_not_cancer,execution_time,comments,status,window_size,stride,matchPercentage,tissuePercentage,id))
        conn.commit()
    except Exception as e:
        raise ValueError('Could not execute function getFolder - Error '+str(e))
    finally:
        conn.close()

def run_image_reader_script(path_Image, path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder, cancer_color, not_cancer_color, patient, path_artifacts_geojson=None):
    command = [
        PYTHON_PATH, IMAGE_READER_PATH,
        '--path_Image', path_Image,
        '--path_cancer_folder', path_cancer_folder,
        '--path_not_cancer_folder', path_not_cancer_folder,
        '--path_cancer_mask_folder', path_cancer_mask_folder,
        '--path_not_cancer_mask_folder', path_not_cancer_mask_folder,
        '--cancer_color', cancer_color,
        '--not_cancer_color', not_cancer_color,
        '--patient', patient
    ]
    if path_artifacts_geojson and os.path.exists(path_artifacts_geojson):
        command.extend(['--path_artifacts_geojson', path_artifacts_geojson])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running {IMAGE_READER_PATH}: {result.stderr}")

    try:
        parsed_output = json.loads(result.stdout)
        status = parsed_output.get("status")
        comments = parsed_output.get("comments")
        cancer_count = parsed_output.get("cancer_patches_created", 0)
        not_cancer_count = parsed_output.get("not_cancer_patches_created", 0)
        return status, comments, cancer_count, not_cancer_count
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to decode JSON from imageReader script. Output: {result.stdout}")

def mainProcess():
    createfolders()
    if LOADCASES:
        AddNewCases()
        print('New images added to the database. Update the colors and re-run the process')
        sys.exit(0)

    totalCases = len(getTotalCases())
    while totalCases > 0:
        data = getCaseToProcess()
        if not data:
            break
        
        case_data = data[0]
        id_cur = case_data['ID']
        path_Image_current = case_data['FILEPATH']
        patient_current = case_data['PATIENT']
        cancer_color_current = case_data['CANCER_COLOR']
        not_cancer_color_current = case_data['NOT_CANCER_COLOR']
        start_time = time.time()

        path_artifacts_geojson_current = None
        if USE_ADVANCED_ARTIFACT_FILTERING:
            geojson_path = os.path.join(os.getenv('GEOJSON_PATH'), os.path.basename(os.path.splitext(path_Image_current)[0] + '.geojson'))
            if os.path.exists(geojson_path):
                path_artifacts_geojson_current = geojson_path
            else:
                print(f"Warning: Artifact filtering is ON, but GeoJSON not found at {geojson_path}. Proceeding without artifact check.")

        try:
            status, comments, count_cancer, count_not_cancer = run_image_reader_script(
                path_Image=path_Image_current,
                path_cancer_folder=PATH_CANCER_FOLDER,
                path_not_cancer_folder=PATH_NOT_CANCER_FOLDER,
                path_cancer_mask_folder=PATH_CANCER_MASK_FOLDER,
                path_not_cancer_mask_folder=PATH_NOT_CANCER_MASK_FOLDER,
                cancer_color=cancer_color_current,
                not_cancer_color=not_cancer_color_current,
                patient=patient_current,
                path_artifacts_geojson=path_artifacts_geojson_current
            )
        except RuntimeError as e:
            status, comments, count_cancer, count_not_cancer = 'FAILED', str(e), 0, 0
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        
        updateCase(id_cur, count_cancer, count_not_cancer, execution_time, status, str(comments[-240:]), WINDOW_SIZE, STRIDE, TISSUE_PERCENTAGE, MATCH_PERCENTAGE)
        
        totalCases -= 1
        print(f"There are {totalCases} images to be processed yet")

if __name__ == '__main__':
    load_dotenv(override=True)
    
    IMAGESPATH = os.getenv('IMAGESPATH')
    PATH_CANCER_FOLDER = os.getenv('PATH_CANCER_FOLDER')
    PATH_NOT_CANCER_FOLDER = os.getenv('PATH_NOT_CANCER_FOLDER')
    PATH_CANCER_MASK_FOLDER = os.getenv('PATH_CANCER_MASK_FOLDER')
    PATH_NOT_CANCER_MASK_FOLDER = os.getenv('PATH_NOT_CANCER_MASK_FOLDER')
    WINDOW_SIZE = os.getenv('WINDOW_SIZE')
    STRIDE = os.getenv('STRIDE')
    MATCH_PERCENTAGE = str(os.getenv('MATCH_PERCENTAGE'))
    TISSUE_PERCENTAGE = str(os.getenv('TISSUE_PERCENTAGE'))
    IMAGE_READER_PATH = str(os.getenv('IMAGE_READER_PATH'))
    PYTHON_PATH = str(os.getenv('PYTHON_PATH'))
    LOADCASES = os.getenv('LOADCASES', 'False').lower() in ('true', '1', 't')
    USE_ADVANCED_ARTIFACT_FILTERING = os.getenv('USE_ADVANCED_ARTIFACT_FILTERING', 'False').lower() in ('true', '1', 't')
    
    mainProcess()