import os
import cv2


def resize_images(input_folder, output_folder, scale_percent=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {filename}")
                continue

            # Compute new dimensions
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * sca le_percent / 100)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            # Save resized image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized {filename} -> {output_path}")


# Define paths
input_folder = "./ADOXYOLO/JPEGImages/"
output_folder = "./ADOXYOLO/Resize/"

resize_images(input_folder, output_folder)