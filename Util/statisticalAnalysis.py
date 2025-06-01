import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.dates as mdates
import cv2

# Paths to images and labels
IMAGE_DIR = "./ADOXYOLO/JPEGImages/"
LABEL_DIR = "./ADOXYOLO/labels/"
CSV_OUTPUT_DIR = "./ADOXYOLO/statistics/"
REPLACE_DATES_FILE = "./replace date.txt"

# Ensure output directory exists
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Define constants
IMAGE_SIZE = (4656, 3496)  # Updated image size
CIRCLE_RADIUS = 200  # Radius in pixels to check for new insects


def parse_filename(filename):
    """Extract location and date from filename."""
    parts = filename.split("_")
    location = parts[0].replace("namhae", "")  # Remove "namhae" prefix
    year, month, day = parts[1:4]
    formatted_date = f"{day}-{month}-{year}"  # Convert to dd-mm-yyyy
    return int(location), formatted_date

def load_interest_points(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points = [list(map(float, line.strip().split())) for line in lines]
    return np.array(points, dtype=np.float32)


def compute_affine_from_files(img1_path, img2_path):
    # Replace .jpg with .txt for point files
    points_dir = "./ADOXYOLO/InterestPoint/Points/"
    file1 = os.path.basename(img1_path).replace('.jpg', '.txt')
    file2 = os.path.basename(img2_path).replace('.jpg', '.txt')
    scale_percent = 100  # Resize to 50% of original size
    scale_factor = scale_percent / 100

    points1_path = os.path.join(points_dir, file1)
    points2_path = os.path.join(points_dir, file2)

    # Load points
    pts1 = load_interest_points(points1_path)
    pts2 = load_interest_points(points2_path)
    print(pts1)
    print(pts2)
    affine_matrix, inliers = cv2.estimateAffine2D( pts1, pts2, method=cv2.RANSAC)

    if affine_matrix is not None:
        H_affine = np.vstack([affine_matrix, [0, 0, 1]])  # Convert to 3x3 homography-like matrix
    else:
        H_affine = np.eye(3)
    return H_affine

def compute_homography_from_files(img1_path, img2_path):
    # Replace .jpg with .txt for point files
    points_dir = "./ADOXYOLO/InterestPoint/Points/"
    file1 = os.path.basename(img1_path).replace('.jpg', '.txt')
    file2 = os.path.basename(img2_path).replace('.jpg', '.txt')
    scale_percent = 100  # Resize to 50% of original size
    scale_factor = scale_percent / 100

    points1_path = os.path.join(points_dir, file1)
    points2_path = os.path.join(points_dir, file2)

    # Load points
    pts1 = load_interest_points(points1_path)
    pts2 = load_interest_points(points2_path)


    """
    # Visualize the points
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_vis = img1.copy()
    img2_vis = img2.copy()
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.circle(img1, pt1, 5, (0, 255, 0), -1)
        cv2.putText(img1, str(i), (pt1[0] + 5, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img2, pt2, 5, (0, 0, 255), -1)
        cv2.putText(img2, str(i), (pt2[0] + 5, pt2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Image 1 with Points", img1)
    cv2.imshow("Image 2 with Points", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    pts1 = pts1 * scale_factor
    pts2 = pts2 * scale_factor
    print(pts1)
    print(pts2)
    # Compute homography
    H, status = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)  # Note: pts2 -> pts1
    # Evaluate accuracy by projecting pts2 to img1
    """
    pts2_proj = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(pts2_proj - pts1, axis=1)
    mean_error = np.mean(errors)
    print("Reprojection Errors:", errors)
    print("Mean Reprojection Error:", mean_error)
    """

    if H is not None:
        H_original = H.copy()

        # Scale translation components
        H_original[0, 2] *= 1 / scale_factor  # a3 adjustment
        H_original[1, 2] *= 1 / scale_factor  # a6 adjustment
        return H_original
    else:
        return np.eye(3)  # Identity if nothing works


def validate_homography(H, pts1, pts2):
    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])

    # Apply homography
    projected_pts2_h = (H @ pts1_h.T).T

    # Convert back from homogeneous
    projected_pts2 = projected_pts2_h[:, :2] / projected_pts2_h[:, 2, np.newaxis]

    # Compute error (Euclidean distance)
    errors = np.linalg.norm(projected_pts2 - pts2, axis=1)

    print("Reprojection Errors:")
    for i, error in enumerate(errors):
        print(f"Point {i + 1}: {error:.2f} pixels")

    print(f"\nMean Error: {np.mean(errors):.2f} pixels")
    return errors

def read_labels(label_file):
    """Reads a YOLO label file and returns a list of insect positions."""
    insect_positions = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == "1":  # Only count insects (ADOX = 1)
                    x_center, y_center = float(parts[1]), float(parts[2])
                    # Convert normalized coordinates to pixel values
                    x_pixel = int(x_center * IMAGE_SIZE[0])
                    y_pixel = int(y_center * IMAGE_SIZE[1])
                    insect_positions.append((x_pixel, y_pixel))
    return insect_positions


def read_replace_dates():
    """Reads the file containing sticky pad replacement dates."""
    replace_dates = defaultdict(set)
    with open(REPLACE_DATES_FILE, 'r') as f:
        for line in f:
            filename = line.strip()
            if filename:
                location, date_taken = parse_filename(filename)
                replace_dates[location].add(date_taken)
    return replace_dates

def find_new_warp(cur_filename, the_first_filename):
    # Load two consecutive images
    img1 = cv2.imread(cur_filename)
    img2 = cv2.imread(the_first_filename)

    # Resize images for visualization (adjust scale as needed)
    scale_percent = 20  # Resize to 50% of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)

    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matching keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    cv2.imshow("Feature Matching", result)
    cv2.waitKey(0)
    """# Warp img2 to align with img1
    height, width = img1.shape[:2]  # Ignore the third channel
    aligned_img2 = cv2.warpPerspective(img2, H, (width, height))"""
    return H

def warp_insect_positions(insect_positions, H):
    """Warp insect positions using homography matrix H."""
    # Convert positions to homogeneous coordinates (x, y) -> (x, y, 1)
    ones = np.ones((len(insect_positions), 1))
    positions_homogeneous = np.hstack([insect_positions, ones])

    # Apply homography transformation
    warped_positions = np.dot(H, positions_homogeneous.T).T

    # Convert back to (x, y) by dividing by the homogeneous coordinate
    warped_positions[:, 0] /= warped_positions[:, 2]
    warped_positions[:, 1] /= warped_positions[:, 2]

    return warped_positions[:, :2]  # Return only (x, y)


def warp_insect_positions_OF(cur_filename, the_first_filename, insect_positions):
    # Load two consecutive images
    img1 = cv2.imread(cur_filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(the_first_filename, cv2.IMREAD_GRAYSCALE)
    scale_percent = 10  # Resize to 50% of original size
    w = int(img1.shape[1] * scale_percent / 100)
    h = int(img1.shape[0] * scale_percent / 100)
    dim = (w, h)

    img1resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)


    flow = cv2.calcOpticalFlowFarneback(img1resized, img2resized, None,
                                        0.5, 3, 50, 3, 5, 1.2, 0)
    # flow = flow * (100 - scale_percent) / 100
    # Convert insect positions to numpy array
    insect_positions = np.array(insect_positions, dtype=np.float32)
    insect_positions = insect_positions * scale_percent / 100

    # Ensure the shape is correct for vectorized operations
    if len(insect_positions.shape) == 1:
        insect_positions = insect_positions.reshape(-1, 2)

    # Extract optical flow vectors at insect positions
    displacement = np.array([flow[int(y), int(x)] for x, y in insect_positions])

    # Update insect positions with flow displacement
    warped_positions = (insect_positions + displacement) * 100 / scale_percent

    # Convert back to list of tuples
    warped_positions = [(int(x), int(y)) for x, y in warped_positions]

    return warped_positions

#------------------------------------Grid matching------------------------
def merge_lines(lines, angle_threshold=10, rho_threshold=20):
    vertical_lines = []
    horizontal_lines = []

    for rho, theta in lines:
        angle_deg = np.rad2deg(theta)  # Convert to degrees
        if abs(angle_deg % 180) < angle_threshold:  # Vertical line
            vertical_lines.append((rho, theta))
        elif abs((angle_deg - 90) % 180) < angle_threshold:  # Horizontal line
            horizontal_lines.append((rho, theta))

    def merge_similar(lines_list):
        """ Merge lines that are close in rho """
        if not lines_list:
            return []
        lines_list.sort()  # Sort by rho
        merged = [lines_list[0]]

        for rho, theta in lines_list[1:]:
            last_rho, last_theta = merged[-1]
            if abs(rho - last_rho) > rho_threshold:
                merged.append((rho, theta))  # Keep only distinct lines

        return merged

    vertical_lines = merge_similar(vertical_lines)
    horizontal_lines = merge_similar(horizontal_lines)

    return vertical_lines + horizontal_lines
"""
def detect_grid_lines_BK(image):
    # Detects grid lines using Canny and Hough Transform.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(image, 30, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    detected_lines = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            detected_lines.append((rho, theta))
    return detected_lines
"""

def detect_grid_lines(image):
    """Detects grid lines using Canny and Hough Transform."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    edges = cv2.Canny(enhanced, 30, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    filtered_lines = []
    angle_tolerance = 20  # Accept lines in the range [-10°, 10°] and [80°, 100°]

    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta)
        if (abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance):
            filtered_lines.append((rho, theta))

    filtered_lines = merge_lines(filtered_lines)
    lines = filtered_lines

    if lines is None:
        print("No lines detected")
        return []

    detected_lines = []
    for line in lines:
        rho, theta = line  # Extract rho and theta properly
        detected_lines.append((rho, theta))
    return detected_lines

def compute_intersections(lines, image_shape):
    """Finds intersections between detected grid lines."""
    intersections = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]

            # Convert from polar to Cartesian (standard line equations)
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])

            # Solve for (x, y)
            if np.linalg.det(A) != 0:  # Ensure the matrix is not singular
                xy = np.linalg.solve(A, b)
                x, y = int(xy[0][0]), int(xy[1][0])

                # Check if the intersection is within the image bounds
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    intersections.append((x, y))

    return intersections


def match_grid_points(intersections1, intersections2):
    """Finds nearest corresponding grid points between two sets of intersections."""
    matched_points = []

    for pt1 in intersections1:
        min_dist = float("inf")
        best_match = None

        for pt2 in intersections2:
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist < min_dist:
                min_dist = dist
                best_match = pt2

        if best_match:
            matched_points.append((pt1, best_match))

    return matched_points

def convert_points_to_keypoints(points):
    """ Convert a list of (x, y) coordinates to OpenCV KeyPoints """
    return [cv2.KeyPoint(float(x), float(y), 1) for x, y in points]

def warp_insect_positions_GridMatching(cur_filename, the_first_filename):
    # Load two consecutive images
    img1 = cv2.imread(cur_filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(the_first_filename, cv2.IMREAD_GRAYSCALE)
    # Resize images for visualization (adjust scale as needed)
    scale_percent = 20  # Resize to 50% of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    scale_factor = scale_percent / 100
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    # Detect grid lines
    lines1 = detect_grid_lines(img1)
    lines2 = detect_grid_lines(img2)
    print("len line1:", len(lines1), ", line2: ", len(lines2))
    if(len(lines1) < 5 or len(lines2) < 5 or (len(lines1) + len(lines2) > 300)):
        H = np.eye(3)  # 3x3 Identity matrix
    else:
        # Compute grid intersections
        intersections1 = compute_intersections(lines1, img1.shape)
        intersections2 = compute_intersections(lines2, img2.shape)
        # Match intersections
        matched_points = match_grid_points(intersections1, intersections2)
        print("matched_points: ", len(matched_points))
        # Extract corresponding points
        src_pts = np.float32([m[1] for m in matched_points])  # Points from img2
        dst_pts = np.float32([m[0] for m in matched_points])  # Points from img1

        # Compute homography using RANSAC

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    H_original = H.copy()

    # Scale translation components
    H_original[0, 2] *= 1 / scale_factor  # a3 adjustment
    H_original[1, 2] *= 1 / scale_factor  # a6 adjustment
    checkfile = "namhae2_2024_5_18_org_1"
    # checkfile = "namhae2_2024_5_20_org_1"
    # checkfile = "namhae2_2024_5_19_org_1"
    # checkfile = "abc"
    # checkfile = "namhae2_2024_11_08_org_1"
    if(checkfile in the_first_filename or (checkfile in cur_filename)):
        # Warp img2 to align with img1
        height, width = img1.shape
        aligned_img2 = cv2.warpPerspective(img2, H, (width, height))
        blend = cv2.addWeighted(img1.astype(np.float32), 0.5, aligned_img2.astype(np.float32), 0.5, 0)
        blend = blend.astype(np.uint8)
        stacked = np.hstack([img1, img2, aligned_img2, blend])
        # Convert to KeyPoints
        kp2 = convert_points_to_keypoints(src_pts)
        kp1 = convert_points_to_keypoints(dst_pts)
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
        cv2.imshow("Blended Alignment Check", stacked)
        cv2.imshow("matched_img", matched_img)
        cv2.waitKey(0)

    return H_original

#--------------------------hidrid insect loaction and hough line transform------------------
def compute_homography_from_points(src_pts, dst_pts):
    if len(src_pts) >= 4 and len(dst_pts) >= 4:
        H, mask = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)

        # Check if H is valid
        if H is None or np.isnan(H).any():
            print("Homography computation failed!")
            return None  # Return identity matrix

        """# Check the number of inliers
        inlier_ratio = np.sum(mask) / len(mask)
        if inlier_ratio < 0.5:  # Less than 50% good matches → unreliable H
            print("Warning: Poor homography estimation. Too many outliers!")
            return np.eye(3)  # Identity transformation
        """

        return H

    print("Not enough points to compute homography.")
    return None  # Identity if no valid transformation


def detect_and_match_insects(img1, img2, insect_positions1, insect_positions2):
    """Compute homography using insect positions, if possible."""
    if len(insect_positions1) >= 4 and len(insect_positions2) >= 4:
        return compute_homography_from_points(insect_positions1, insect_positions2)
    return None  # Not enough insects


def warp_images_based_on_insects_or_grid(cur_filename, pre_filename):
    # Load two consecutive images
    img1 = cv2.imread(cur_filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(pre_filename, cv2.IMREAD_GRAYSCALE)
    # Resize images for visualization (adjust scale as needed)
    scale_percent = 20  # Resize to 50% of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    scale_factor = scale_percent / 100
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    label_file1 = os.path.join(LABEL_DIR, cur_filename.replace(".jpg", ".txt"))
    insect_positions1 = read_labels(label_file1)

    label_file2 = os.path.join(LABEL_DIR, pre_filename.replace(".jpg", ".txt"))
    insect_positions2 = read_labels(label_file2)

    H = detect_and_match_insects(img1, img2, insect_positions1, insect_positions2)

    if H is None:  # Not enough insects, use grid
        # Detect grid lines
        lines1 = detect_grid_lines(img1)
        lines2 = detect_grid_lines(img2)
        print("len line1:", len(lines1), ", line2: ", len(lines2))
        if (len(lines1) < 5 or len(lines2) < 5 or (len(lines1) + len(lines2) > 300)):
            H = np.eye(3)  # 3x3 Identity matrix
        else:
            # Compute grid intersections
            intersections1 = compute_intersections(lines1, img1.shape)
            intersections2 = compute_intersections(lines2, img2.shape)
            # Match intersections
            matched_points = match_grid_points(intersections1, intersections2)
            print("matched_points: ", len(matched_points))
            # Extract corresponding points
            src_pts = np.float32([m[1] for m in matched_points])  # Points from img2
            dst_pts = np.float32([m[0] for m in matched_points])  # Points from img1

            # Compute homography using RANSAC

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if H is not None:
        H_original = H.copy()

        # Scale translation components
        H_original[0, 2] *= 1 / scale_factor  # a3 adjustment
        H_original[1, 2] *= 1 / scale_factor  # a6 adjustment
        return H_original
    else:
        return np.eye(3)  # Identity if nothing works
"""
def load_interest_points(POINTS_DIR, image_name):
    # Load interest points (9 points in 3x3 grid) from file.
    points_file = os.path.join(POINTS_DIR, image_name.replace('.jpg', '.txt'))
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            points = [list(map(float, line.strip().split())) for line in f]
        return np.array(points, dtype=np.float32)
    return None
"""
def compute_gt_theta(POINTS_DIR, im_name, target_name, scale_factor = 0.1):
    """ Compute GT_theta (2x3 affine transformation matrix) from interest points. """
    src_pts = self.load_interest_points(POINTS_DIR, im_name)  # 9 points from the source image
    dst_pts = self.load_interest_points(POINTS_DIR, target_name)  # 9 points from the target image

    src_pts = src_pts * scale_factor
    dst_pts = dst_pts * scale_factor

    if src_pts is None or dst_pts is None or len(src_pts) != 9 or len(dst_pts) != 9:
        raise ValueError(f"Invalid interest points for {im_name} or {target_name}")

    A = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)  # shape (18, 9)

    # Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]  # last row of V^T (smallest singular value)
    H = h.reshape((3, 3))

    return H / H[-1, -1]  # normalize

def warp_point(H, pt):
    pt_h = np.array([pt[0], pt[1], 1.0])
    warped = H @ pt_h
    return warped[:2] / warped[2]

def warp_insect_positions(insect_positions, H):
    ones = np.ones((len(insect_positions), 1))
    positions_homogeneous = np.hstack([insect_positions, ones])
    warped_positions = np.dot(H, positions_homogeneous.T).T
    warped_positions[:, 0] /= warped_positions[:, 2]
    warped_positions[:, 1] /= warped_positions[:, 2]
    return warped_positions[:, :2]

def process_data():
    """Processes images and labels, then saves statistics."""
    location_data = defaultdict(list)
    image_records = []
    replace_dates = read_replace_dates() #list of replace date of each location

    # Read and sort images by location first, then by date
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg"):
            parsed = parse_filename(filename)
            if parsed:
                location, date_taken = parsed
                image_records.append((location, date_taken, filename))

    # Sort by location first, then by date
    image_records.sort(key=lambda x: (x[0], datetime.strptime(x[1], "%d-%m-%Y"))) #sort (location, date, file name) by location then date taken

    # Process images in sorted order
    new_insect_count = 0
    previous_positions = set()
    cur_location = -1
    reference_frame_file = image_records[0][2]
    H_previous = np.eye(3)  # 3x3 Identity matrix

    for location, date_taken, filename in image_records:
        # print(filename)
        label_file = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))
        insect_positions = read_labels(label_file)


        # Initialize the count of new insects for a new location

        if(location != cur_location):
            new_insect_count = 0
            previous_positions = set()  # Reset positions if sticky pad is replaced
            cur_location = location
            reference_frame_file = filename
            H_previous = np.eye(3)  # 3x3 Identity matrix
        # Check for sticky pad replacement and reset positions if necessary
        if date_taken in replace_dates[location]:
            previous_positions = set()  # Reset positions if sticky pad is replaced
            # new_insect_count = 0
            reference_frame_file = filename
            H_previous = np.eye(3)  # 3x3 Identity matrix
        else:
            """for record in location_data[location]:
                _, _, _, prev_pos = record  # Unpack previous insect positions
                previous_positions.update(prev_pos)
            """
        print(reference_frame_file, ":::", filename)

        if(len(insect_positions) != 0 ):
            # MatchingMode = "ORB"
            # MatchingMode = "OF"
            #MatchingMode = "GRIDMATCHING"
            MatchingMode = "INTERESTPOINTS"

            if(MatchingMode == "ORB"):
                H_cur = find_new_warp(os.path.join(IMAGE_DIR, reference_frame_file), os.path.join(IMAGE_DIR, filename ))
                H_previous = H_previous * H_cur
                insect_positions_warp = warp_insect_positions(insect_positions, H_previous)
                insect_positions_warp = [(int(x), int(y)) for x, y in insect_positions_warp]
            elif(MatchingMode == "OF"):
                insect_positions_warp = warp_insect_positions_OF(os.path.join(IMAGE_DIR, reference_frame_file), os.path.join(IMAGE_DIR, filename ), insect_positions)
            elif(MatchingMode == "GRIDMATCHING"):
                H_cur = warp_insect_positions_GridMatching(os.path.join(IMAGE_DIR, reference_frame_file),
                                                                 os.path.join(IMAGE_DIR, filename))
                H_previous = H_previous * H_cur
                insect_positions_warp = warp_insect_positions(insect_positions, H_previous)
                insect_positions_warp = [(int(x), int(y)) for x, y in insect_positions_warp]
            elif (MatchingMode == "HYBRID"):
                H_cur = warp_images_based_on_insects_or_grid(os.path.join(IMAGE_DIR, reference_frame_file),
                                                           os.path.join(IMAGE_DIR, filename))
                H_previous = H_previous * H_cur
                insect_positions_warp = warp_insect_positions(insect_positions, H_previous)
                insect_positions_warp = [(int(x), int(y)) for x, y in insect_positions_warp]
            elif (MatchingMode == "INTERESTPOINTS"):
                H_previous = compute_homography_from_files(os.path.join(IMAGE_DIR, filename),
                                                           os.path.join(IMAGE_DIR, reference_frame_file))
                print(H_previous)
                if isinstance(insect_positions, list):
                    insect_positions = np.array(insect_positions, dtype=np.float32)

                if insect_positions.ndim == 2 and insect_positions.shape[1] == 2:
                    insect_positions_warp = cv2.perspectiveTransform(
                        insect_positions.reshape(-1, 1, 2), H_previous
                    ).reshape(-1, 2)
                else:
                    insect_positions_warp = np.empty((0, 2), dtype=np.float32)  # Handle empty or invalid input
                # insect_positions_warp = warp_insect_positions(insect_positions, H_previous)
                # insect_positions_warp = [(int(x), int(y)) for x, y in insect_positions_warp]
            # Count new insects based on positions
            print("insect_positions: ", insect_positions, "_insect_positions_warp: ", insect_positions_warp, ": ", np.linalg.norm(np.array(insect_positions) - np.array(insect_positions_warp)))
            new_positions = []
            for pos in insect_positions_warp:
                if not any(np.linalg.norm(np.array(pos) - np.array(prev_pos)) < CIRCLE_RADIUS for prev_pos in
                           previous_positions):
                    new_insect_count += 1  # Count as new insect
                    new_positions.append(pos)
                    previous_positions.add(tuple(pos))
        checkfile = "namhae2_2024_5_18_org_1"
        checkfile = "namhae2_2024_5_18_org_1"
        # checkfile = "namhae2_2024_11_08_org_1"
        # checkfile = "abc"
        if (checkfile in reference_frame_file or (checkfile in filename) or True ):
            img1 = cv2.imread(os.path.join(IMAGE_DIR, reference_frame_file), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(IMAGE_DIR, filename), cv2.IMREAD_GRAYSCALE)
            # Resize images for visualization
            height, width = img2.shape[:2]
            aligned_img2 = cv2.warpPerspective(img2, H_previous, (width, height))
            scale_percent = 20  # Resize to 50% of original size
            width = int(img1.shape[1] * scale_percent / 100)
            height = int(img1.shape[0] * scale_percent / 100)
            dim = (width, height)

            img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
            aligned_img2 = cv2.resize(aligned_img2, dim, interpolation=cv2.INTER_AREA)

            # Warp img2 to align with img1
            height, width = img1.shape

            blend = cv2.addWeighted(img1.astype(np.float32), 0.5, aligned_img2.astype(np.float32), 0.5, 0)
            blend = blend.astype(np.uint8)

            blendori = cv2.addWeighted(img1.astype(np.float32), 0.5, img2.astype(np.float32), 0.5, 0)
            blendori = blendori.astype(np.uint8)

            blendimg2 = cv2.addWeighted(img2.astype(np.float32), 0.5, aligned_img2.astype(np.float32), 0.5, 0)
            blendimg2 = blendimg2.astype(np.uint8)
            # img_viz = cv2.imread(os.path.join(IMAGE_DIR, reference_frame_file))

            # Assuming previous_positions is a list of (x, y) tuples
            print(previous_positions)
            for idx, (x, y) in enumerate(previous_positions):
                cv2.circle(img1, (int(x/5), int(y/5)), radius=5, color=(255, 0, 255), thickness=1)  # Red filled circle
                cv2.putText(img1, str(idx), (int(x/5) + 10, int(y/5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255 , 255), 1, cv2.LINE_AA)  # White text next to the circle

            # cv2.imshow("Previous Positions", img_viz)
            stacked = np.hstack([img1, blendori, blendimg2, blend])
            # Convert to KeyPoints
            cv2.imshow("Blended Alignment Check", stacked)
            cv2.imshow("aligned_img2", aligned_img2)
            cv2.waitKey(0)
        # Store the current image data, including the new insect count
        print((date_taken, filename, new_insect_count, previous_positions))
        location_data[location].append((date_taken, filename, new_insect_count, previous_positions))
        # reference_frame_file = filename

    # Process each location and save statistics
    for location, records in location_data.items():
        df = pd.DataFrame(records, columns=["Date Taken", "Image Name", "New Insect Count", "Insect Positions"])
        df = df.drop(columns=["Insect Positions"])  # Remove positions column for CSV
        df.sort_values(by="Date Taken", inplace=True)

        # Convert date to datetime object
        df["Date Taken"] = pd.to_datetime(df["Date Taken"], format="%d-%m-%Y")

        # Fill in missing dates with the previous day's insect countb
        date_range = pd.date_range(start=df["Date Taken"].min(), end=df["Date Taken"].max())
        df = df.set_index("Date Taken").reindex(date_range).fillna(method="ffill").reset_index()
        df.rename(columns={"index": "Date Taken"}, inplace=True)

        # Save to CSV
        csv_file = os.path.join(CSV_OUTPUT_DIR, f"location_{location}.csv")
        df.to_csv(csv_file, index=False)

        # Plot graph for each location
        plt.figure(figsize=(10, 5))
        plt.plot(df["Date Taken"], df["New Insect Count"], marker='o', linestyle='-', label="New Insect Count")

        # Mark sticky pad replacement dates
        for date in replace_dates[location]:
            plt.axvline(x=pd.to_datetime(date, format="%d-%m-%Y"), color='r', linestyle='--',
                        label='Sticky Pad Replaced')

        plt.xlabel("Date Taken")
        plt.ylabel("Number of New Insects")
        plt.title(f"New Insects Over Time - Location {location}")

        # Reduce date label density
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Show one tick per week
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

        plt.grid()
        plt.tight_layout()

        # Avoid overlapping labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for i, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[i])
        plt.legend(unique_handles, unique_labels)

        plt.savefig(os.path.join(CSV_OUTPUT_DIR, f"location_{location}.png"))
        plt.show()


if __name__ == "__main__":
    process_data()
