import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
from four_point import four_point_transform
from pytesseract import image_to_string
from sys import exit, stderr

image = cv2.imread('img.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()

# Resizing
image = imutils.resize(image, height=500)

# Greyscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smoothing (Blur); Helps reduce noise
smooth = cv2.GaussianBlur(grey, (5, 5), 0)

# Edge-detection
edged = cv2.Canny(grey, 75, 200)

# Contours
cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sorting contours in the descending order of sizes
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find the document page
screen_contour = None
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screen_contour = approx
		break

# Drawing new contours
if screen_contour is not None:
	cv2.drawContours(image, [screen_contour], -1, (0, 255, 0), 2)
	cv2.imshow('Drawing', image)
	cv2.waitKey(0)
else:
	exit(0)

# Bird-eye view
warped = four_point_transform(orig, screen_contour.reshape(4, 2) * ratio)
cv2.imshow('Bird-eye view', warped)
cv2.waitKey(0)

# Recognize and display text
try:
	text = image_to_string(warped)
	print(str(text))
except Exception as e:
	print(e, file=stderr)