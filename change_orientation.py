from pytesseract import Output
import pytesseract
import argparse
import imutils
from skimage.segmentation import clear_border
from imutils import contours
import cv2
import numpy as np


def rotate_text(image):
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

	# rotate the image to correct the orientation
	rotated = imutils.rotate_bound(image, angle=results["rotate"])
	return rotated

def crop_image(img):
# Convert the rotated image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply thresholding or any other preprocessing as needed
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	# Find contours
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Find the largest contour
	largest_contour = max(contours, key=cv2.contourArea)


    # Calculate angles of detected lines  
	# Get bounding rectangle of the largest contour
	x, y, w, h = cv2.boundingRect(largest_contour)

	# Draw bounding rectangle on the original image
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Crop the image based on the bounding rectangle of the largest contour
	cropped_image = img[y:y+h, x:x+w]
	return cropped_image

def find_tilt_angle(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # print(angle)
        angles.append(angle)

    # Check if any angles are not within the threshold range for horizontal lines
    threshold_range = 10  # Adjust as needed
    horizontal_angles = [angle for angle in angles if -threshold_range <= angle <= threshold_range]

    # If there are non-horizontal lines, calculate the average angle
    if not horizontal_angles == angles:
        median_angle = np.median(angles)
    else:
        median_angle = 0
        # print("hi")

    return -1 * median_angle


def general_pipeline(image):

    angle = find_tilt_angle(image)
    rotated_image = imutils.rotate_bound(image, angle)

    return rotated_image

def detect_text(img, word):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # # Apply thresholding
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     blurred = cv2.medianBlur(gray, 5)
     thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
     thresh = cv2.bitwise_not(thresh)
     custom_config = r'--oem 3 --psm 11'
     data = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=custom_config)
    
	# Define the minimum confidence and minimum bounding box size
     min_confidence = 0
     min_width = 30  # Minimum width of the bounding box
     min_height = 20  # Minimum height of the bounding box
     for i in range(len(data['text'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data["text"][i]
        conf = int(data["conf"][i])
    
        if conf > min_confidence and w > min_width and h > min_height:
            print(text, conf)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if word in text:
                 box_coords = [x, y, w, h]
                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
     return box_coords



def find_line(img, date_bbox):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection (using Canny edge detector)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Apply Hough Transform to detect lines (Probabilistic Hough Transform)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=400, maxLineGap=300)
    
    # Initialize variables for the closest line
    closest_line = None
    min_distance = float('inf')
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line is approximately horizontal
            if np.abs(y2 - y1) < 20:  # Adjust this threshold as needed
                # Calculate the y-coordinate of the bottom of the line
                bottom_y = (y1 + y2) // 2
                
                # Calculate the distance to the bottom of the "date" bounding box
                distance = np.abs(bottom_y - (date_bbox[1] + date_bbox[3]))
                
                # Check if the x-range of the line includes the x-range of the "date" bounding box
                if date_bbox[0] >= x1 and date_bbox[0] + date_bbox[2] <= x2:
                    # Check if this line is closer than the current closest line
                    if distance < min_distance:
                        min_distance = distance
                        closest_line = (x1, y1, x2, y2)
    
    # Draw the closest horizontal line on the image
    if closest_line is not None:
        x1, y1, x2, y2 = closest_line
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    # grab the internal Python iterator for the list of character
    # contours, then initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []
    
    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character contour from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)
                
                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)
                
                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))
        except StopIteration:
            break
    
    # return a tuple of the ROIs and locations
    return (rois, locs)



             



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="/Users/katiewang/Desktop/IMG_1599.jpg")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
rotated = rotate_text(image)
corrected_image = general_pipeline(rotated)

cropped = crop_image(corrected_image)
date_bbox = detect_text(cropped, "DATE")


# Find the closest horizontal line
line_image = find_line(cropped, date_bbox)



cv2.imshow('img', line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Display the corrected image
# cv2.imshow("Bounding Rectangle", rotated)
# cv2.imshow("Corrected Image", corrected_image)
# cv2.imshow("Cropped Image", cropped)


