import cv2
import os
import numpy as np

# Directories
IMAGE_DIR = "./ADOXYOLO/JPEGImages/"
SAVE_DIR = "./ADOXYOLO/InterestPoint/"
POINTS_DIR = "./ADOXYOLO/InterestPoint/Points/"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(POINTS_DIR, exist_ok=True)

# Global variables
points = []
current_image = None
current_filename = ""
scale_factor = 0.2  # Resize factor


def click_event(event, x, y, flags, param):
    global points, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        point_id = len(points) + 1
        points.append((int(x / scale_factor), int(y / scale_factor)))  # Scale back to original size

        # Draw circle and point ID
        cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(current_image, str(point_id), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Image", current_image)


def save_annotations():
    global points, current_filename
    if points:
        # Save points to a text file
        txt_filename = os.path.join(POINTS_DIR, current_filename.replace(".jpg", ".txt"))
        with open(txt_filename, "w") as f:
            for x, y in points:
                f.write(f"{x} {y}\n")
        print(f"Saved: {txt_filename}")

        # Save the annotated image
        save_path = os.path.join(SAVE_DIR, current_filename)
        cv2.imwrite(save_path, current_image)
        print(f"Saved: {save_path}")

def draw_points(img, points):
    """Draw interest points with numbered labels."""

    newPoints = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]  # ✅ Correct way
    for idx, (x, y) in enumerate(newPoints):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def load_points(image_name):
    """Load interest points from a text file."""
    points_file = os.path.join(POINTS_DIR, image_name.replace('.jpg', '.txt'))
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            return [tuple(map(int, line.strip().split())) for line in f]
    return []

# Load images from folder
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
image_files.sort()
flag = True
index = 0
for img_file in image_files:
    index = index + 1
    print(index, ": ", img_file)
    if("namhae2_2024_9_23_org_1" in img_file):
        flag = True
    if(flag == False):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    resized_image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    current_image = resized_image.copy()
    current_filename = img_file
    points = load_points(current_filename)

    draw_points(current_image, points)
    cv2.imshow("Image", current_image)
    cv2.setMouseCallback("Image", click_event)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):  # Save
            save_annotations()
        elif key == ord("n"):  # Next image

            break
        elif key == ord('r'):  # Reset points
            print("Reset Points: ", current_filename)
            points = []  # Reset points for each image


cv2.destroyAllWindows()