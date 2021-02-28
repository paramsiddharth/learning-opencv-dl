import cv2
import imutils

# Reading the image in greyscale
image = cv2.imread('demo.png')
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Greyscale', grey)
cv2.waitKey(3000)

# Edge-detection
edged = cv2.Canny(grey, 30, 150)
cv2.imshow('Edged', edged)
cv2.waitKey(3000)

# Thresholding
thresh = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('Thresholded', thresh)
cv2.waitKey(3000)

# Finding contours
cntr = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntr = imutils.grab_contours(cntr)
print(f'Number of shapes: {len(cntr)}')