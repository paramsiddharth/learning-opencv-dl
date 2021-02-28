# Imports
import imutils
import cv2

# Read image
image = cv2.imread('demo.jpg')

# Height, Width, and Depth
h, w, d = image.shape

# Colours
B, G, R = image[187, 200] # This is the standard order in OpenCV

# Extracting a region of interest
roi = image[60:160, 360:420]

# Display the image
cv2.imshow('ROI', roi)
cv2.waitKey(3000) # Waits for a key to be pressed, else holds for 3 seconds

# Resize the image
resized = imutils.resize(image, width=300) # Will scale the height to match proportion
cv2.imshow('Resized', resized)
cv2.waitKey(3000)

# Rotate the image
rotated = imutils.rotate(image, 45)
cv2.imshow('Rotated', rotated)
cv2.waitKey(3000)

# Gaussian Blur
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow('Blurred', blurred)
cv2.waitKey(3000)