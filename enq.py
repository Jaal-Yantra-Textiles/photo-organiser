import argparse
import queue
import threading
import os
from single  import process_single_face_image, cluster_and_organize_images
import convert
import sample


job_queue = queue.Queue()

def worker():
    while True:
        task, args = job_queue.get()
        if task is None:  # sentinel to stop the worker
            break
        task(*args)
        job_queue.task_done()

def process_images(single_face_folder):
    single_face_image_files = [os.path.join(single_face_folder, f) for f in os.listdir(single_face_folder) if f.endswith('.JPEG')]
    print(f"Total single face images to process: {len(single_face_image_files)}")
    for img_file in single_face_image_files:
        job_queue.put((process_single_face_image, [img_file]))

def cluster_images():
    job_queue.put((cluster_and_organize_images, []))

def start_sample():
    job_queue.put((sample.train_model, []))

def enqueue_conversion(single_face_folder):
    single_face_image_files = [os.path.join(single_face_folder, f) for f in os.listdir(single_face_folder) if f.endswith('.JPEG')]
    for img_file in single_face_image_files:
        job_queue.put((convert.convert_to_jpeg, [img_file]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and organize images using tensorflow and python Queue.")
    subparsers = parser.add_subparsers(dest="command")

    # Command for processing images
    process_parser = subparsers.add_parser("process", help="Process images.")
    process_parser.add_argument("folder", help="Path to the folder containing images to be processed.")

    # Command for clustering and organizing images
    cluster_parser = subparsers.add_parser("cluster", help="Cluster and organize images.")

    # Command for converting images
    convert_parser = subparsers.add_parser("convert", help="Convert .JPG images to .JPEG format.")
    convert_parser.add_argument("folder", help="Path to the folder containing images to be converted.")

    # Command for sample model
    convert_parser = subparsers.add_parser("sample", help="Run a sample model for testing the GPU capability.")
    

    args = parser.parse_args()

    num_worker_threads = 4  # adjust as needed
    for _ in range(num_worker_threads):
        threading.Thread(target=worker).start()

    if args.command == "process":
        process_images(args.folder)
    elif args.command == "cluster":
        cluster_images()
    elif args.command == "convert":
        enqueue_conversion(args.folder)  
    elif args.command == "sample":
        start_sample()  

    # Wait for all tasks to complete
    job_queue.join()

    # Stop the workers
    for _ in range(num_worker_threads):
        job_queue.put((None, []))
