import argparse
import queue
import threading
import os
from cluster import cluster_and_organize_images
from convert import convert_to_jpeg

from logger_util import logger
from time import sleep

from sqlite_utility import SQLiteDB
from migrations import MigrationUtility

job_queue = queue.Queue()

def worker():
    while True:
        task, args = job_queue.get()
        if task is None:  # sentinel to stop the worker
            break
        task(*args)
        job_queue.task_done()
        sleep(1)

def process_images(folder=None, process_type=None):
    from core import compare_all_images, extract_and_save_features, detect_and_store_face, extract_features_from_stored_roi, compare_all_images_stats
    images = None
    # Check if folder is provided and exists
    if not folder or not os.path.exists(folder):
        logger.warning(f"Folder not provided must be without a folder processing type: {process_type}")
    else:
        # Get the list of images from the folder
        images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.JPG')]
        logger.info(f"Total single face images to process: {len(images)}")
    
    # If there are no images to process, log a warning and exit
    if not images:
        logger.warning(f"No '.JPG' images found in the provided folder: {folder}")
        
    
    if process_type == 'face':
        job_queue.put((detect_and_store_face, [folder]))
    elif process_type == 'features':
        job_queue.put((extract_and_save_features, [folder]))
    elif process_type == 'storedroi':
        job_queue.put((extract_features_from_stored_roi, []))
    elif process_type == 'compare':
        job_queue.put((compare_all_images, []))
    elif process_type == 'statscompare':
        job_queue.put((compare_all_images_stats, []))
    else:
        logger.warning(f"Invalid process type provided: {process_type}")
        raise ValueError(f"Invalid process type: {process_type}")


def output_stats():
    from core import generate_scatter_plot_image
    generate_scatter_plot_image()

def cluster_images():
    job_queue.put((cluster_and_organize_images, []))

def start_sample():
    import sample
    job_queue.put((sample.train_model, []))

def enqueue_conversion(single_face_folder, output_folder, resize):
    single_face_image_files = [os.path.join(single_face_folder, f) for f in os.listdir(single_face_folder) if f.endswith('.JPG')]
    
    total_images = len(single_face_image_files)
    for idx, img_file in enumerate(single_face_image_files, 1):
        converted_file = convert_to_jpeg(img_file, output_folder, resize)
        if converted_file:
            logger.info(f"Converted {img_file} to {converted_file} ({idx}/{total_images})")
        else:
            logger.warning(f"Failed to convert {img_file} ({idx}/{total_images})")


def initialize_db():
    """
    Initialize or re-initialize the database.
    """
    db = SQLiteDB("images_comparison.db")
    db.initialize_database()


def migrate_db(db_path, migration):
    """
    Migration on the database.
    """
    logger.info("Applying Migrations")
    migrator = MigrationUtility(db_path)

    if not migrator.is_migrations_table_present():
        logger.info("Migrations Table Does not Exist Creating One")
        migrator.create_migrations_table()

    applied_migrations = migrator.list_applied_migrations()

    logger.info(applied_migrations)

    if migration:
        if migration not in applied_migrations:
            migrator.apply_migration(migration)
            logger.info(f"Applied migration: {migration}")
        else:
            logger.warn(f"Migration {migration} has already been applied and will not be executed again.")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and organize images using tensorflow and python Queue.")
    subparsers = parser.add_subparsers(dest="command")

    # Command for processing images
    process_parser = subparsers.add_parser("process", help="Process images.")
    process_parser.add_argument("folder", nargs='?', default=None, help="Path to the folder containing images to be processed.")
    process_parser.add_argument("--type", help="Process with command  like face, features, storedroi, compare, statscompare")
    # Command for clustering and organizing images
    cluster_parser = subparsers.add_parser("cluster", help="Cluster and organize images.")

    # Command for converting images
    convert_parser = subparsers.add_parser("convert", help="Convert .JPG images to .JPEG format.")
    convert_parser.add_argument("folder", help="Path to the folder containing images to be converted.")
    convert_parser.add_argument("output_folder", help="Path to the folder where the output converted images will be stored like /path/newfolder.")
    convert_parser.add_argument("--resize",  action="store_true",  help="Whether to resize images to 636x1024 or not. Default is False.")

    # Command for sample model
    convert_parser = subparsers.add_parser("sample", help="Run a sample model for testing the GPU capability.")
    compare_parser = subparsers.add_parser("compare", help="Compare features of all images with each other.")

    init_db_parser = subparsers.add_parser("initdb", help="Initialize or re-initialize the database.")

    migrate_parser = subparsers.add_parser("migrate", help="Database Migration Utility")
    migrate_parser.add_argument("--db_path", required=True, help="Path to the SQLite database file.")
    migrate_parser.add_argument("--migration", help="Path to the migration SQL script.")


    stats_image_parser = subparsers.add_parser("statsimage", help="Output the stats image")

    


    args = parser.parse_args()

    num_worker_threads = 4  # adjust as needed
    for _ in range(num_worker_threads):
        threading.Thread(target=worker).start()

    if args.command == "process":
        process_images(args.folder, args.type)
    elif args.command == "cluster":
        cluster_images()
    elif args.command == "convert":
        enqueue_conversion(args.folder, args.output_folder, args.resize)  
    elif args.command == "sample":
        start_sample()
    elif args.command == "initdb":
        initialize_db()
    elif args.command == "migrate":
        migrate_db(args.db_path, args.migration)
    elif args.command == "statsimage":
        output_stats()
        

    # Wait for all tasks to complete
    job_queue.join()

    # Stop the workers
    for _ in range(num_worker_threads):
        job_queue.put((None, []))
