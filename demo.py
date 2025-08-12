import cv2
import numpy as np

# Load image
image = cv2.imread("Dataset/DB1/archive/Bacterialblight/BACTERAILBLIGHT3_001.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for thresholding (example: brown/yellow disease spots)
lower = np.array([10, 50, 50])     # adjust for your case
upper = np.array([30, 255, 255])   # adjust for your case

# Thresholding
mask = cv2.inRange(hsv, lower, upper)

# Apply mask to extract ROI
result = cv2.bitwise_and(image, image, mask=mask)

# Save or display result
cv2.imwrite("thresholded_roi.jpg", result)
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("ROI", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
