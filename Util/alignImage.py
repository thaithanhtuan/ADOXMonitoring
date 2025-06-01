import cv2
import numpy as np

# Load two consecutive images
img1 = cv2.imread("./ADOXYOLO/JPEGImages/namhae2_2024_5_18_org_1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./ADOXYOLO/JPEGImages/namhae2_2024_5_19_org_1.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("./ADOXYOLO/JPEGImages/namhae2_2024_11_09_org_1.jpg", cv2.IMREAD_GRAYSCALE)

# Resize images for visualization (adjust scale as needed)
scale_percent = 20  # Resize to 50% of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

def merge_lines(lines, angle_tolerance=2, rho_tolerance=20):
    merged = []
    for rho, theta in lines:
        if not merged or all(abs(rho - m[0]) > rho_tolerance or abs(theta - m[1]) > np.radians(angle_tolerance) for m in merged):
            merged.append((rho, theta))
    return merged

def detect_grid_lines(image):
    """Detects grid lines using Canny and Hough Transform."""
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # imageF = cv2.filter2D(image, -1, kernel)
    # imageF2 = cv2.filter2D(enhanced, -1, kernel)
    edges = cv2.Canny(enhanced, 30, 150, apertureSize=3)
    # kernel = np.ones((3, 3), np.uint8)
    # edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    filtered_lines = []
    angle_tolerance = 10  # Accept lines in the range [-10°, 10°] and [80°, 100°]

    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta)
        if (abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance):
            filtered_lines.append((rho, theta))
    """
    stacked = np.hstack([image, enhanced, imageF,imageF2])
    cv2.imshow("Blended Alignment Check", stacked)
    cv2.waitKey(0)
    """
    # filtered_lines = merge_lines(filtered_lines)
    print(len(filtered_lines))
    img_viz = image.copy()
    # vizimg = np.ones(image.shape) * 255
    for line in filtered_lines:
        if len(line) == 2:  # (rho, theta) case
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))  # Extend line
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
        else:  # (x1, y1, x2, y2) case
            x1, y1, x2, y2 = line[0]

        cv2.line(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    stacked = np.hstack([image, enhanced, img_viz, edges])
    cv2.imshow("Blended Alignment Check", stacked)
    cv2.waitKey(0)


    detected_lines = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            detected_lines.append((rho, theta))

    return detected_lines

lines2 = detect_grid_lines(img2)

def detect_grid_lines(image):
    """Detects grid lines using Canny and Hough Transform."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    detected_lines = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            detected_lines.append((rho, theta))

    return detected_lines


def compute_optical_flow(prev_gray, cur_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 25, 3, 5, 1.2, 0)
    return flow

def find_global_motion(prev_gray, cur_gray):
    flow = compute_optical_flow(prev_gray, cur_gray)

    # Compute median flow (assumes most of the motion is camera shift)
    median_flow = np.median(flow.reshape(-1, 2), axis=0)

    # Create transformation matrix (Translation only)
    H = np.array([[1, 0, median_flow[0]],
                  [0, 1, median_flow[1]],
                  [0, 0, 1]], dtype=np.float32)

    return H

def warp_image(image, H):
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, H, (width, height))


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

# Detect grid lines
lines1 = detect_grid_lines(img1)
lines2 = detect_grid_lines(img2)
# Compute grid intersections
intersections1 = compute_intersections(lines1, img1.shape)
intersections2 = compute_intersections(lines2, img2.shape)
# Match intersections
matched_points = match_grid_points(intersections1, intersections2)

# Extract corresponding points
src_pts = np.float32([m[1] for m in matched_points])  # Points from img2
dst_pts = np.float32([m[0] for m in matched_points])  # Points from img1

# Compute homography using RANSAC
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp img2 to align with img1
height, width = img1.shape
aligned_img2 = cv2.warpPerspective(img2, H, (width, height))

# Show the aligned image
stacked = np.hstack([img1, img2, aligned_img2])

blend = cv2.addWeighted(img1.astype(np.float32), 0.5, aligned_img2.astype(np.float32), 0.5, 0)

# Show the blended image
cv2.imshow("Blended Alignment Check", blend.astype(np.uint8))

cv2.imshow("Original Image 1 | Original Image 2 | Aligned Image", stacked)
cv2.waitKey(0)

# Compute transformation and warp
prev_gray = img2
cur_gray = img1
H = find_global_motion(prev_gray, cur_gray)
aligned_image = warp_image(cur_gray, H)

grid_lines = detect_grid_lines(prev_gray)
cv2.imshow("Grid Lines", grid_lines)
cv2.waitKey(0)
cv2.imshow("Aligned Image", aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
