import pymysql
import os
import re
import subprocess
import time
import json
from datetime import datetime
import sys
import argparse
import shutil


def open_conn():
    conn = pymysql.connect(
        charset="utf8mb4",
        connect_timeout=30,
        cursorclass=pymysql.cursors.DictCursor,
        db="defaultdb",
        host="mysql-1ced8c6-bruno-bc2b.b.aivencloud.com",
        password="AVNS_0fHWcE7eJC6LeXyJFHB",
        read_timeout=60,
        port=11025,
        user="avnadmin",
        write_timeout=60,
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
        # --- NOVOS CAMINHOS LOCAIS (TEMPORÁRIOS) ---
        LOCAL_WSI_DIR = '/content/temp_wsi'
        LOCAL_PATCHES_CANCER_DIR = '/content/temp_patches/cancer'
        LOCAL_MASKS_CANCER_DIR = '/content/temp_patches/cancer_mask'
        LOCAL_PATCHES_NOTCANCER_DIR = '/content/temp_patches/not_cancer'
        LOCAL_MASKS_NOTCANCER_DIR = '/content/temp_patches/not_cancer_mask'
        
        # Cria os diretórios locais uma vez
        os.makedirs(LOCAL_WSI_DIR, exist_ok=True)
        os.makedirs(LOCAL_PATCHES_CANCER_DIR, exist_ok=True)
        os.makedirs(LOCAL_MASKS_CANCER_DIR, exist_ok=True)
        os.makedirs(LOCAL_PATCHES_NOTCANCER_DIR, exist_ok=True)
        os.makedirs(LOCAL_MASKS_NOTCANCER_DIR, exist_ok=True)

        stop=False
        totalCases=len(getTotalCases())
        while stop==False:
            data=getCaseToProcess()
            if len(data)==0:
                stop=True
                continue

            # Obtém os caminhos do Google Drive a partir do banco de dados
            id_cur=data[0]['ID']
            drive_wsi_path = data[0]['FILEPATH']
            
            # Paths permanentes no Drive para onde os patches serão movidos
            drive_cancer_patch_path = PATH_CANCER_FOLDER
            drive_not_cancer_patch_path = PATH_NOT_CANCER_FOLDER
            drive_cancer_mask_path = PATH_CANCER_MASK_FOLDER
            drive_not_cancer_mask_path = PATH_NOT_CANCER_MASK_FOLDER
            
            # Parâmetros
            cancer_color_current=data[0]['CANCER_COLOR']
            not_cancer_color_current=data[0]['NOT_CANCER_COLOR']
            patient_current=data[0]['PATIENT']

            local_wsi_filepath = '' # Para uso no bloco finally
            
            try:
                # --- ETAPA 1: COPIAR ARQUIVOS PARA O AMBIENTE LOCAL ---
                print(f"Copying {os.path.basename(drive_wsi_path)} to local runtime...")
                local_wsi_filepath = os.path.join(LOCAL_WSI_DIR, os.path.basename(drive_wsi_path))
                shutil.copy(drive_wsi_path, local_wsi_filepath)

                drive_ndpa_path = drive_wsi_path + '.ndpa'
                drive_ndpa_path = drive_ndpa_path.replace('/IMAGES/', '/ANNOTATIONS/')
                local_ndpa_filepath = local_wsi_filepath + '.ndpa'
                if os.path.exists(drive_ndpa_path):
                    shutil.copy(drive_ndpa_path, local_ndpa_filepath)
                
                print("Copy complete. Starting processing...")
                start_time = time.time()
                
                # --- ETAPA 2: EXECUTAR O PROCESSAMENTO USANDO OS CAMINHOS LOCAIS ---
                status, comments = run_image_reader_script(
                    path_Image=local_wsi_filepath, # <<< USA CAMINHO LOCAL
                    path_cancer_folder=LOCAL_PATCHES_CANCER_DIR, # <<< USA CAMINHO LOCAL
                    path_not_cancer_folder=LOCAL_PATCHES_NOTCANCER_DIR, # <<< USA CAMINHO LOCAL
                    path_cancer_mask_folder=LOCAL_MASKS_CANCER_DIR, # <<< USA CAMINHO LOCAL
                    path_not_cancer_mask_folder=LOCAL_MASKS_NOTCANCER_DIR, # <<< USA CAMINHO LOCAL
                    cancer_color=cancer_color_current,
                    not_cancer_color=not_cancer_color_current,
                    patient=patient_current
                )

                end_time = time.time()
                execution_time = (end_time - start_time)/60
                
                # Contagem dos patches gerados LOCALMENTE
                count_cancer = len(os.listdir(LOCAL_PATCHES_CANCER_DIR))
                count_not_cancer = len(os.listdir(LOCAL_PATCHES_NOTCANCER_DIR))
                
                # Atualiza o banco de dados com o resultado
                updateCase(id_cur, count_cancer, count_not_cancer, execution_time, status, str(comments[-240:]),WINDOW_SIZE,STRIDE,TISSUE_PERCENTAGE,MATCH_PERCENTAGE)
                
                # --- ETAPA 3: MOVER OS PATCHES GERADOS DE VOLTA PARA O GOOGLE DRIVE ---
                print(f"Moving {count_cancer} cancer patches and {count_not_cancer} non-cancer patches to Google Drive...")
                for f in os.listdir(LOCAL_PATCHES_CANCER_DIR):
                    shutil.move(os.path.join(LOCAL_PATCHES_CANCER_DIR, f), os.path.join(drive_cancer_patch_path, f))
                for f in os.listdir(LOCAL_PATCHES_NOTCANCER_DIR):
                    shutil.move(os.path.join(LOCAL_PATCHES_NOTCANCER_DIR, f), os.path.join(drive_not_cancer_patch_path, f))
                for f in os.listdir(LOCAL_MASKS_CANCER_DIR):
                    shutil.move(os.path.join(LOCAL_MASKS_CANCER_DIR, f), os.path.join(drive_cancer_mask_path, f))
                for f in os.listdir(LOCAL_MASKS_NOTCANCER_DIR):
                    shutil.move(os.path.join(LOCAL_MASKS_NOTCANCER_DIR, f), os.path.join(drive_not_cancer_mask_path, f))

                totalCases=totalCases-1
                print(f"Move complete. {totalCases} images remaining.")

            except Exception as e:
                # Se ocorrer um erro, registra e continua para a próxima imagem
                status = 'FAILED_IN_PIPELINE'
                comments = f"Error in mainProcess loop: {e}"
                updateCase(id_cur, 0, 0, 0, status, str(comments[-240:]), WINDOW_SIZE, STRIDE, TISSUE_PERCENTAGE, MATCH_PERCENTAGE)
                continue
            
            finally:
                # --- ETAPA 4: LIMPEZA DO AMBIENTE LOCAL (CRUCIAL!) ---
                # Garante que os arquivos locais sejam limpos, mesmo se ocorrer um erro.
                print("Cleaning up local runtime environment...")
                if os.path.exists(local_wsi_filepath):
                    os.remove(local_wsi_filepath)
                if os.path.exists(local_wsi_filepath + '.ndpa'):
                    os.remove(local_wsi_filepath + '.ndpa')
                
                # Esvazia os diretórios de patches para a próxima iteração
                for d in [LOCAL_PATCHES_CANCER_DIR, LOCAL_MASKS_CANCER_DIR, LOCAL_PATCHES_NOTCANCER_DIR, LOCAL_MASKS_NOTCANCER_DIR]:
                    for f in os.listdir(d):
                        os.remove(os.path.join(d, f))
                print("Cleanup complete.")

if __name__ == '__main__':
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

    parser = argparse.ArgumentParser(description='Extract patches and masks from WSI based on annotations.')
    parser.add_argument('--loadcases', action='store_true', help='Load cases')

    args = parser.parse_args()

    IMAGESPATH = "/content/drive/MyDrive/DOWNLOADS_DIAGSET/IMAGES"
    PATH_CANCER_FOLDER = "/content/drive/MyDrive/DOWNLOADS_DIAGSET/PATCHES/CANCER"
    PATH_NOT_CANCER_FOLDER = "/content/drive/MyDrive/DOWNLOADS_DIAGSET/PATCHES/NOT_CANCER"
    PATH_CANCER_MASK_FOLDER = "/content/drive/MyDrive/DOWNLOADS_DIAGSET/PATCHES/CANCER_MASK"
    PATH_NOT_CANCER_MASK_FOLDER = "/content/drive/MyDrive/DOWNLOADS_DIAGSET/PATCHES/NOT_CANCER_MASK"
    WINDOW_SIZE = "224"
    STRIDE = "112"
    MATCH_PERCENTAGE = "0.9"
    TISSUE_PERCENTAGE = "0.3"
    IMAGE_READER_PATH = "/content/2_imageReader.py"
    PYTHON_PATH = "/usr/bin/python3"
    LOADCASES = args.loadcases

    mainProcess()