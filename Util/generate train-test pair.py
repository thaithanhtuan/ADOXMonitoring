import os
import random
import re
from datetime import datetime

# Paths
image_folder = "./ADOXYOLO/JPEGImages/"
replace_file = "./replace date.txt"
train_file = "train.txt"
test_file = "test.txt"
REPLACE_DATES_FILE = "./replace date.txt"
IMAGE_DIR = "./ADOXYOLO/JPEGImages/"
LABEL_DIR = "./ADOXYOLO/labels/"

def parse_filename(filename):
    """Extract location and date from filename."""
    parts = filename.split("_")
    location = parts[0].replace("namhae", "")  # Remove "namhae" prefix
    year, month, day = parts[1:4]
    formatted_date = f"{day}-{month}-{year}"  # Convert to dd-mm-yyyy
    return int(location), formatted_date


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

# Function to extract date from filename
def extract_date(filename):
    match = re.search(r"(\d{4})_(\d{1,2})_(\d{1,2})", filename)  # YYYY_MM_DD
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))  # (Year, Month, Day)
    return (0, 0, 0)

from collections import defaultdict
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

# Group images by location and replace period
grouped_images = {}
period = 0
cur_location = -1

for (location, date_taken, filename) in image_records:
    if (location != cur_location):  # If location changes, reset period
        period = 0
        cur_location = location

    # If the date matches a replacement date for this location, increase period
    if date_taken in replace_dates[location]:
        period = period + 1

    # Add image to grouped_images under the correct location and period
    if location not in grouped_images:
        grouped_images[location] = {}

    if period not in grouped_images[location]:
        grouped_images[location][period] = []

    grouped_images[location][period].append(filename)

pairs = set()  # Use a set to avoid duplicates

for location, periods in grouped_images.items():
    for period, images in periods.items():
        if len(images) < 2:
            continue  # Skip if fewer than two images

        random.shuffle(images)  # Shuffle images in place

        # Create consecutive pairs (img1, img2), (img2, img3), ...
        for i in range(len(images) - 1):
            pair = (images[i], images[i + 1])
            pairs.add(pair)  # Add to set (avoiding duplicates)

pairs = list(pairs)  # Convert set to list before shuffling
random.shuffle(pairs)  # Now shuffle is safe

split_idx = int(0.8 * len(pairs))
train_pairs = pairs[:split_idx]
test_pairs = pairs[split_idx:]

# Write to train.txt
with open(train_file, "w") as f:
    for pair in train_pairs:
        f.write(" ".join(pair) + "\n")  # Convert tuple to a space-separated string

with open(test_file, "w") as f:
    for pair in test_pairs:
        f.write(" ".join(pair) + "\n")  # Convert tuple to a space-separated string

print(f"Generated {train_file} ({len(train_pairs)} pairs) and {test_file} ({len(test_pairs)} pairs).")
