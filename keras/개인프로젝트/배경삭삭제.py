import cv2
# rembg 패키지에서 remove 클래스 불러오기
from rembg import remove 

# PIL 패키지에서 Image 클래스 불러오기
from PIL import Image 
path = "./asian_data/2/"
# Load the image
image = cv2.imread(path + "00011A02.jpg")

# Convert to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color threshold
lower_threshold = (0, 0, 0)
upper_threshold = (179, 255, 50)

# Create a binary mask based on the threshold
mask = cv2.inRange(hsv, lower_threshold, upper_threshold)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Save the resulting image
cv2.imwrite(path + "00011A02_remove1.jpg", result)

