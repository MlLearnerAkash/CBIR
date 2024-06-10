# import os

# def files_name(root_path):
#     list_files= []
#     for dir, dirs, files in os.walk(root_path):
#         for file in files:
#             file_path = os.path.join(file, dir)
#             list_files.append(file_path)
#     return list_files

# if __name__=="__main__":
#     root_path = "/home/pravaig-20/Downloads/Assignment_CVML_02_04_24/Assignment/datasets/RESISC45_partial/"

#     print(files_name(root_path))

import os
from PIL import Image

def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

def find_images(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image(file_path):
                image_files.append(file_path)
    return image_files

def write_to_file(image_files, output_file):
    with open(output_file, 'w') as f:
        for image in image_files:
            f.write(image + '\n')

if __name__ == "__main__":
    directory_to_search = "/home/pravaig-20/Downloads/Assignment_CVML_02_04_24/Assignment/datasets/RESISC45_partial/"  # Replace with your directory path
    output_file = "image_files.txt"  # Output file path

    images = find_images(directory_to_search)
    write_to_file(images, output_file)
    print(f"Found {len(images)} image files. Paths are written to {output_file}.")

