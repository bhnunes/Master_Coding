import sqlite3
import os
import subprocess
import time
import json
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# --- NEW: Style class for attractive printing ---
class Style:
    """A helper class for styling terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    
    # Emojis for status indicators
    INFO = 'ℹ️'
    SUCCESS = '✅'
    WARNING = '⚠️'
    ERROR = '❌'
    ROCKET = '🚀'
    DB = '📦'
    FOLDER = '📁'
    CHECK = '🔍'

def open_conn(db_path):
    """Opens a connection to the SQLite database."""
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            print(f"{Style.INFO} Database directory not found. Creating '{db_dir}'...")
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"{Style.RED}{Style.ERROR} Fatal error connecting to database at '{db_path}': {e}{Style.RESET}")
        sys.exit(1)

def create_folders_and_db_table(tag, base_path, db_path):
    """Handles the initial setup of folders and the database table."""
    images_folder = os.path.join(base_path, f"IMAGES_{tag}")
    annotations_folder = os.path.join(base_path, f"ANNOTATIONS_{tag}")

    if not os.path.exists(images_folder) or not os.path.exists(annotations_folder):
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(annotations_folder, exist_ok=True)
        print(f"\n{Style.BLUE}{Style.BOLD}--- PROJECT SETUP ---{Style.RESET}")
        print(f"{Style.GREEN}{Style.SUCCESS} '{images_folder}' created.{Style.RESET}")
        print(f"{Style.GREEN}{Style.SUCCESS} '{annotations_folder}' created.{Style.RESET}")
        print(f"{Style.YELLOW}{Style.INFO} Please move your images and annotations to these folders.{Style.RESET}")
        print(f"{Style.BOLD}   IMPORTANT: Image and annotation files must share the same base name (e.g., case01.svs and case01.xml).{Style.RESET}")
        sys.exit(0)
    
    table_name = f"DATABASE_{tag}"
    conn = open_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if cursor.fetchone():
            print(f"{Style.INFO} Table '{Style.CYAN}{table_name}{Style.RESET}' already exists.")
        else:
            print(f"{Style.INFO} Creating new table: '{Style.CYAN}{table_name}{Style.RESET}'...")
            cursor.execute(f"""
            CREATE TABLE {table_name} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT, STATUS TEXT DEFAULT 'TO BE PROCESSED',
                IMAGEPATH TEXT NOT NULL, ANNOTATIONPATH TEXT, LastUpdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CANCER_QTD INTEGER, NON_CANCER_QTD INTEGER, VALID_IMAGE INTEGER,
                CANCER_COLOR TEXT, NOT_CANCER_COLOR TEXT, PROCESSINGTIME_MINUTES REAL,
                PATIENT TEXT NOT NULL UNIQUE, COMMENTS TEXT, WINDOW_SIZE INTEGER, STRIDE INTEGER,
                MATCH_PERCENTAGE TEXT, TISSUE_PERCENTAGE TEXT
            )""")
            conn.commit()
            print(f"{Style.GREEN}{Style.SUCCESS} Table created successfully.{Style.RESET}")
    except sqlite3.Error as e:
        raise ValueError(f"Could not create database table - Error: {e}")
    finally:
        conn.close()

def ingest_new_cases(tag, base_path, db_path, activate_sanity_check, use_advanced_filtering, geojson_path): # MODIFIED: Added new args
    """
    Scans folders, adds new images, performs GeoJSON sanity check if enabled,
    and reports if SVS files were added.
    """
    images_folder = os.path.join(base_path, f"IMAGES_{tag}")
    annotations_folder = os.path.join(base_path, f"ANNOTATIONS_{tag}")
    table_name = f"DATABASE_{tag}"
    svs_files_added = False

    if not os.listdir(images_folder):
        raise FileNotFoundError(f"The directory '{images_folder}' is empty. Please add images to process.")

    print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.CHECK} INGESTION PROCESS ---{Style.RESET}")
    conn = open_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT IMAGEPATH, PATIENT FROM {table_name}")
        existing_data = cursor.fetchall()
        existing_basenames = {os.path.basename(row['IMAGEPATH']) for row in existing_data}
        existing_patients = {int(row['PATIENT']) for row in existing_data if row['PATIENT'].isdigit()}

        annotation_lookup = {os.path.splitext(f)[0]: os.path.join(annotations_folder, f) for f in os.listdir(annotations_folder)}
        image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
        new_cases_to_add = []

        # --- NEW: GEOJSON Sanity Check Setup ---
        geojson_basenames = set()
        run_geojson_check = activate_sanity_check and use_advanced_filtering
        if run_geojson_check:
            print(f"{Style.INFO} GeoJSON sanity check is {Style.GREEN}ACTIVE{Style.RESET}.")
            if not geojson_path or not os.path.isdir(geojson_path):
                raise FileNotFoundError(f"GeoJSON sanity check is active, but GEOJSON_PATH ('{geojson_path}') is invalid.")
            geojson_basenames = {os.path.splitext(f)[0] for f in os.listdir(geojson_path) if f.lower().endswith('.geojson')}
            print(f"{Style.INFO} Found {len(geojson_basenames)} GeoJSON files for comparison.")
        # --- END NEW ---

        for image_file in tqdm(image_files, desc=f"{Style.CYAN}Scanning for new images{Style.RESET}"):
            if image_file in existing_basenames:
                continue
            
            image_path = os.path.join(images_folder, image_file)
            if image_path.lower().endswith('.svs'):
                svs_files_added = True

            image_basename = os.path.splitext(image_file)[0]
            annotation_path = annotation_lookup.get(image_basename)
            
            status, comments = 'TO BE PROCESSED', ''
            
            if not annotation_path:
                status, comments = 'FAILED', 'The equivalent annotation file could not be found.'

            # --- NEW: Apply GeoJSON Sanity Check ---
            if status == 'TO BE PROCESSED' and run_geojson_check:
                if image_basename not in geojson_basenames:
                    status = 'FAILED'
                    comments = 'GeoJSON Sanity Check Failed: The equivalent GeoJSON file was not found.'
            # --- END NEW ---
            
            next_patient_id = max(existing_patients) + 1 if existing_patients else 100001
            patient_id_str = str(next_patient_id)
            existing_patients.add(next_patient_id)
            
            new_cases_to_add.append((image_path, annotation_path, patient_id_str, status, comments))
        
        if new_cases_to_add:
            print(f"{Style.GREEN}{Style.SUCCESS} Found {len(new_cases_to_add)} new cases to add to the database.{Style.RESET}")
            cursor.executemany(f"INSERT INTO {table_name} (IMAGEPATH, ANNOTATIONPATH, PATIENT, STATUS, COMMENTS) VALUES (?, ?, ?, ?, ?)", new_cases_to_add)
            conn.commit()
        else:
            print(f"{Style.INFO} No new image files found to add.")
            
    except sqlite3.Error as e:
        raise ValueError(f'Could not execute function ingest_new_cases - Error {e}')
    finally:
        conn.close()
    
    return svs_files_added


def get_cases_to_process(db_path, table_name):
    conn = open_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE STATUS = 'TO BE PROCESSED'")
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise ValueError(f'Could not execute get_cases_to_process - Error {e}')
    finally:
        conn.close()

def update_case(db_path, table_name, case_id, data):
    conn = open_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""UPDATE {table_name} SET 
                CANCER_QTD = ?, NON_CANCER_QTD = ?, PROCESSINGTIME_MINUTES = ?,
                COMMENTS = ?, STATUS = ?, WINDOW_SIZE = ?, STRIDE = ?,
                MATCH_PERCENTAGE = ?, TISSUE_PERCENTAGE = ?, LastUpdate = CURRENT_TIMESTAMP
            WHERE ID = ?""",
            (
                data['cancer_qtd'], data['non_cancer_qtd'], data['exec_time'],
                data['comments'], data['status'], data['window_size'],
                data['stride'], data['match_percentage'], data['tissue_percentage'],
                case_id
            )
        )
        conn.commit()
    except sqlite3.Error as e:
        tqdm.write(f"Error updating case {case_id} in database: {e}")
    finally:
        conn.close()

def run_image_reader_script(args):
    """Executes the image reader script as a subprocess."""
    command = [
        args['python_path'], args['image_reader_path'],
        '--handler', 'SVS' if args['image_path'].lower().endswith('.svs') else 'NDPI',
        '--path_Image', args['image_path'],
        '--annotation_path', args['annotation_path'],
        '--path_cancer_folder', args['cancer_folder'],
        '--path_not_cancer_folder', args['not_cancer_folder'],
        '--path_cancer_mask_folder', args['cancer_mask_folder'],
        '--path_not_cancer_mask_folder', args['not_cancer_mask_folder'],
        '--cancer_color', args['cancer_color'],
        '--not_cancer_color', args['not_cancer_color'],
        '--patient', args['patient']
    ]
    if args.get('artifacts_geojson_path') and os.path.exists(args['artifacts_geojson_path']):
        command.extend(['--path_artifacts_geojson', args['artifacts_geojson_path']])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        error_message = f"Error running {args['image_reader_path']}: {result.stderr or result.stdout}"
        return {'status': 'FAILED', 'comments': error_message, 'cancer_patches_created': 0, 'not_cancer_patches_created': 0}
        
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        error_message = f"Failed to decode JSON from imageReader. Output: {result.stdout}"
        return {'status': 'FAILED', 'comments': error_message, 'cancer_patches_created': 0, 'not_cancer_patches_created': 0}

def main_process():
    """Main orchestration function."""
    load_dotenv(override=True)
    
    tag = os.getenv('TAG')
    if not tag:
        raise ValueError("The 'TAG' environment variable is not set. Please define it in your .env file.")

    config = {
        'tag': tag, 'db_path': os.path.normpath(os.getenv('SQLITE_DB_PATH')),
        'base_path': os.path.normpath(os.getenv('PROJECTS_BASE_PATH', './projects')),
        'window_size': os.getenv('WINDOW_SIZE'), 'stride': os.getenv('STRIDE'),
        'match_percentage': os.getenv('MATCH_PERCENTAGE'), 'tissue_percentage': os.getenv('TISSUE_PERCENTAGE'),
        'image_reader_path': os.path.normpath(os.getenv('IMAGE_READER_PATH')), 'python_path': os.path.normpath(os.getenv('PYTHON_PATH')),
        'load_cases': os.getenv('LOADCASES', 'False').lower() in ('true', '1', 't'),
        'use_advanced_artifact_filtering': os.getenv('USE_ADVANCED_ARTIFACT_FILTERING', 'False').lower() in ('true', '1', 't'),
        # --- NEW: Read GeoJSON config here ---
        'activate_sanity_check_geojson': os.getenv('ACTIVATE_SANITY_CHECK_GEOJSON', 'False').lower() in ('true', '1', 't'),
        'geojson_path': os.path.normpath(os.getenv('GEOJSON_PATH'))
    }
    table_name = f"DATABASE_{config['tag']}"
    
    create_folders_and_db_table(config['tag'], config['base_path'], config['db_path'])

    if config['load_cases']:
        # --- MODIFIED: Pass new config to the function ---
        svs_added = ingest_new_cases(
            config['tag'], config['base_path'], config['db_path'],
            config['activate_sanity_check_geojson'], 
            config['use_advanced_artifact_filtering'], 
            config['geojson_path']
        )

        print(f"\n{Style.GREEN}{Style.SUCCESS} Ingestion complete. Set {Style.BOLD}LOADCASES=False{Style.RESET}{Style.GREEN} in .env to start processing.{Style.RESET}")
        
        # --- NEW: SVS Color Warning ---
        if svs_added:
            print(f"\n{Style.YELLOW}{Style.BOLD}╔═══════════════════════════════════════════════════════════════════╗")
            print(f"║ {Style.WARNING}  ACTION REQUIRED: SVS Annotation Colors                      {Style.WARNING}  ║")
            print(f"╠═══════════════════════════════════════════════════════════════════╣")
            print(f"║ You have added .svs files, which require manual color setup.      ║")
            print(f"║ Please run a SQL UPDATE query on your database to set the         ║")
            print(f"║ 'CANCER_COLOR' and 'NOT_CANCER_COLOR' for each new .svs file.     ║")
            print(f"║                                                                   ║")
            print(f"║ {Style.CYAN}Example Query:{Style.RESET}{Style.YELLOW}                                                    ║")
            print(f"║ {Style.CYAN}UPDATE {table_name} SET CANCER_COLOR = '65280' WHERE ...;{Style.RESET}{Style.YELLOW}       ║")
            print(f"╚═══════════════════════════════════════════════════════════════════╝{Style.RESET}")
        return

    # Main processing loop
    cases_to_process = get_cases_to_process(config['db_path'], table_name)
    if not cases_to_process:
        print(f"\n{Style.INFO} No cases to process with status 'TO BE PROCESSED'.")
        return
        
    # --- NEW: SVS Color Safeguard ---
    problematic_svs_files = []
    for case in cases_to_process:
        is_svs = case['IMAGEPATH'].lower().endswith('.svs')
        colors_missing = not case['CANCER_COLOR'] or not case['NOT_CANCER_COLOR']
        if is_svs and colors_missing:
            problematic_svs_files.append(os.path.basename(case['IMAGEPATH']))
    
    if problematic_svs_files:
        error_message = (
            f"\n{Style.RED}{Style.ERROR}{Style.BOLD} PROCESSING HALTED: Missing SVS annotation colors.{Style.RESET}\n"
            f"{Style.YELLOW}The following .svs files are marked 'TO BE PROCESSED' but do not have 'CANCER_COLOR' and/or 'NOT_CANCER_COLOR' set in the database.{Style.RESET}\n\n"
            + "\n".join([f"  - {fname}" for fname in problematic_svs_files]) +
            f"\n\nPlease run an UPDATE query on the '{Style.CYAN}{table_name}{Style.RESET}' table to set these values before proceeding."
        )
        raise ValueError(error_message)

    print(f"\n{Style.BLUE}{Style.BOLD}--- {Style.ROCKET} STARTING PROCESSING ---{Style.RESET}")
    patch_base_path = os.path.join(config['base_path'], 'PATCHES')
    path_cancer_folder = os.path.join(patch_base_path, 'CANCER')
    path_not_cancer_folder = os.path.join(patch_base_path, 'NOT_CANCER')
    path_cancer_mask_folder = os.path.join(patch_base_path, 'CANCER_MASK')
    path_not_cancer_mask_folder = os.path.join(patch_base_path, 'NOT_CANCER_MASK')
    for p in [path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder]:
        os.makedirs(p, exist_ok=True)

    with tqdm(total=len(cases_to_process), desc=f"{Style.CYAN}Processing WSI slides{Style.RESET}") as pbar:
        for case in cases_to_process:
            slide_basename = os.path.basename(case['IMAGEPATH'])
            pbar.set_description(f"Processing: {Style.CYAN}{slide_basename}{Style.RESET}")
            
            if not case['ANNOTATIONPATH']:
                pbar.update(1)
                continue

            start_time = time.time()
            
            artifacts_geojson = None
            if config['use_advanced_artifact_filtering'] and config['geojson_path']:
                geojson_file = os.path.join(config['geojson_path'], os.path.splitext(slide_basename)[0] + '.geojson')
                if os.path.exists(geojson_file):
                    artifacts_geojson = geojson_file

            result = run_image_reader_script({
                **config, 'image_path': case['IMAGEPATH'], 'annotation_path': case['ANNOTATIONPATH'],
                'patient': case['PATIENT'], 'cancer_color': case['CANCER_COLOR'],
                'not_cancer_color': case['NOT_CANCER_COLOR'], 'cancer_folder': path_cancer_folder,
                'not_cancer_folder': path_not_cancer_folder, 'cancer_mask_folder': path_cancer_mask_folder,
                'not_cancer_mask_folder': path_not_cancer_mask_folder, 'artifacts_geojson_path': artifacts_geojson
            })
            
            end_time = time.time()
            update_data = {
                'cancer_qtd': result.get('cancer_patches_created', 0), 'non_cancer_qtd': result.get('not_cancer_patches_created', 0),
                'exec_time': (end_time - start_time) / 60, 'comments': str(result.get('comments', ''))[-240:],
                'status': result.get('status', 'FAILED'), 'window_size': config['window_size'], 'stride': config['stride'],
                'match_percentage': config['match_percentage'], 'tissue_percentage': config['tissue_percentage']
            }
            update_case(config['db_path'], table_name, case['ID'], update_data)
            pbar.update(1)
            pbar.set_postfix(cancer=update_data['cancer_qtd'], non_cancer=update_data['non_cancer_qtd'], status=update_data['status'])

if __name__ == '__main__':
    main_process()