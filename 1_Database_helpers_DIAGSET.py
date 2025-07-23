import pymysql
from dotenv import load_dotenv
import os
import re
import subprocess
import time
import json
from datetime import datetime
import sys


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


def find_wsi_files(directory):
    """MODIFICADO: Procura por arquivos .ndpi em vez de .svs."""
    wsi_files = []
    patients=[]
    for root, dirs, files in os.walk(directory):
        for name in files:
            # ALTERAÇÃO PRINCIPAL AQUI
            if name.endswith('.ndpi'):
                file_path = os.path.join(root, name)
                wsi_files.append(file_path)
                # Assumindo que o ID do paciente ainda pode ser extraído do nome da pasta
                patients.append(extract_patient(os.path.basename(file_path).replace('.ndpi', '')))

    return wsi_files, patients


def AddNewCases():
    """MODIFICADO: Chama a nova função para encontrar arquivos .ndpi."""
    wsi_files, patients = find_wsi_files(IMAGESPATH) # Alterado de find_svs_files
    for image, patient in zip(wsi_files, patients):
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

def run_image_reader_script(path_Image, path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder, cancer_color, not_cancer_color, patient): #added mask folders
    command = [
        PYTHON_PATH,
        IMAGE_READER_PATH,
        '--path_Image', path_Image,
        '--path_cancer_folder', path_cancer_folder,
        '--path_not_cancer_folder', path_not_cancer_folder,
        '--path_cancer_mask_folder', path_cancer_mask_folder,  # Add cancer mask folder
        '--path_not_cancer_mask_folder', path_not_cancer_mask_folder, # Add not cancer mask folder
        '--cancer_color', cancer_color,
        '--not_cancer_color', not_cancer_color,
        '--patient', patient
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running {IMAGE_READER_PATH}: {result.stderr}")

    output_json = result.stdout
    parsed_output = json.loads(output_json)
    status = parsed_output.get("status")
    comments = parsed_output.get("comments")

    return status, comments


def count_png_files_in_range(start_time, end_time, folder_path):
    # Convert timestamps to datetime objects
    start_datetime = datetime.fromtimestamp(int(start_time))
    end_datetime = datetime.fromtimestamp(int(end_time))

    png_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                # Get the creation time of the file
                creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                # Check if the file was created within the specified range
                if start_datetime <= creation_time <= end_datetime:
                    png_count += 1
    return png_count


def countTotal(start_time, end_time, folder):
    count = count_png_files_in_range(start_time, end_time, folder)
    return count 


def mainProcess():
    if LOADCASES==True:
        AddNewCases()
        print('New images added to the database. Update the colors and re-run the process')
        sys.exit(0)
    else:
        stop=False
        totalCases=len(getTotalCases())
        while stop==False:
            data=getCaseToProcess()
            if len(data)==0:
                stop=True
            else:
                id_cur=data[0]['ID']
                path_Image_current=data[0]['FILEPATH']
                path_cancer_folder_current=PATH_CANCER_FOLDER
                path_not_cancer_folder_current=PATH_NOT_CANCER_FOLDER
                path_cancer_mask_folder_current = PATH_CANCER_MASK_FOLDER  # Use separate mask folders
                path_not_cancer_mask_folder_current = PATH_NOT_CANCER_MASK_FOLDER
                cancer_color_current=data[0]['CANCER_COLOR']
                not_cancer_color_current=data[0]['NOT_CANCER_COLOR']
                patient_current=data[0]['PATIENT']
                start_time = time.time()

                try:
                    status, comments = run_image_reader_script( # Pass mask folders
                        path_Image=path_Image_current,
                        path_cancer_folder=path_cancer_folder_current,
                        path_not_cancer_folder=path_not_cancer_folder_current,
                        path_cancer_mask_folder=path_cancer_mask_folder_current,
                        path_not_cancer_mask_folder=path_not_cancer_mask_folder_current,
                        cancer_color=cancer_color_current,
                        not_cancer_color=not_cancer_color_current,
                        patient=patient_current
                    )
                except RuntimeError as e:
                    status = 'FAILED'
                    updateCase(id_cur, 0, 0, 0, status, str(comments[-240:]), WINDOW_SIZE, STRIDE, TISSUE_PERCENTAGE, MATCH_PERCENTAGE)
                    continue 


                end_time = time.time()
                execution_time = (end_time - start_time)/60
                count_cancer=countTotal(start_time, end_time,PATH_CANCER_FOLDER )
                count_not_cancer=countTotal(start_time, end_time,PATH_NOT_CANCER_FOLDER)
                updateCase(id_cur, count_cancer, count_not_cancer, execution_time, status, str(comments[-240:]),WINDOW_SIZE,STRIDE,TISSUE_PERCENTAGE,MATCH_PERCENTAGE)
                totalCases=totalCases-1
                print("There are "+str(totalCases)+" images to be processed yet")

if __name__ == '__main__':
    load_dotenv(override=True)
    global IMAGESPATH
    global PATH_CANCER_FOLDER
    global PATH_NOT_CANCER_FOLDER
    global PATH_CANCER_MASK_FOLDER  # Separate mask folders
    global PATH_NOT_CANCER_MASK_FOLDER
    global WINDOW_SIZE
    global STRIDE
    global MATCH_PERCENTAGE
    global TISSUE_PERCENTAGE
    global LOADCASES
    global IMAGE_READER_PATH
    global PYTHON_PATH

    IMAGESPATH = os.getenv('IMAGESPATH')
    PATH_CANCER_FOLDER = os.getenv('PATH_CANCER_FOLDER')
    PATH_NOT_CANCER_FOLDER = os.getenv('PATH_NOT_CANCER_FOLDER')
    PATH_CANCER_MASK_FOLDER = os.getenv('PATH_CANCER_MASK_FOLDER')  # From .env
    PATH_NOT_CANCER_MASK_FOLDER = os.getenv('PATH_NOT_CANCER_MASK_FOLDER')  # From .env
    WINDOW_SIZE = os.getenv('WINDOW_SIZE')
    STRIDE = os.getenv('STRIDE')
    MATCH_PERCENTAGE = str(os.getenv('MATCH_PERCENTAGE'))
    TISSUE_PERCENTAGE = str(os.getenv('TISSUE_PERCENTAGE'))
    IMAGE_READER_PATH = str(os.getenv('IMAGE_READER_PATH'))
    PYTHON_PATH = str(os.getenv('PYTHON_PATH'))
    LOADCASES = False # True to add cases

    mainProcess()