import cv2
import os
import numpy as np

IMAGE_DIR = "./ADOXYOLO/JPEGImages/"
OUTPUT_DIR = "./ADOXYOLO/InterestPoint/"
POINTS_DIR = "./ADOXYOLO/InterestPoint/Points/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(POINTS_DIR, exist_ok=True)

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
current_idx = 0
points = []


def load_points(image_name):
    """Load interest points from a text file."""
    points_file = os.path.join(POINTS_DIR, image_name.replace('.jpg', '.txt'))
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            return [tuple(map(int, line.strip().split())) for line in f]
    return []


def save_points(image_name, points):
    """Save interest points to a text file, scaling them back to original size."""
    points_file = os.path.join(POINTS_DIR, image_name.replace('.jpg', '.txt'))
    with open(points_file, 'w') as f:
        for x, y in points:
            f.write(f"{x * 2} {y * 2}\n")




def draw_points(img, points):
    """Draw interest points with numbered labels."""
    for idx, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def click_event(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        img_display = img.copy()
        img_display = draw_points(img_display, points)
        cv2.imshow("Image", img_display)


def process_image():
    global img, img_display, points

    image_name = image_files[current_idx]


    img = cv2.imread(os.path.join(IMAGE_DIR, image_name))
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Resize to half
    points = load_points(image_name)
    img_display = img.copy()
    img_display = draw_points(img_display, points)
    cv2.imshow("Image", img_display)


process_image()
cv2.setMouseCallback("Image", click_event)

while True:
    print(current_idx, ": ", image_name)
    if ("namhae2_2024_9_15_org_1" in image_name):
        flag = True
    if (flag == False):
        continue
    key = cv2.waitKey(0) & 0xFF

    if key == ord('s'):  # Save points
        save_points(image_files[current_idx], points)
        cv2.imwrite(os.path.join(OUTPUT_DIR, image_files[current_idx]), img_display)
        print(f"Saved: {image_files[current_idx]}")

    elif key == ord('n'):  # Next image
        current_idx += 1
        if current_idx >= len(image_files):
            break
        process_image()

    elif key == ord('r'):  # Reset points
        points = []
        process_image()

    elif key == 27:  # Escape key to exit
        break

cv2.destroyAllWindows()
