# import the numpy package
import numpy as np
# rembg 패키지에서 remove 클래스 불러오기
from rembg import remove 

# PIL 패키지에서 Image 클래스 불러오기
from PIL import Image 
path = "./asian_data/2/"
# Load the image
img = Image.open(path + "00011A02.jpg")

# Remove the background
out = remove(img)

# Convert the output image to RGBA mode
out = out.convert('RGBA')

# Create a NumPy array from the image
arr = np.array(out)

# Set the alpha channel to 0 for pixels below a threshold value (in this case, 100)
arr[(arr[:,:,0] < 100) & (arr[:,:,1] < 100) & (arr[:,:,2] < 100)] = [0, 0, 0, 0]

# Create a new Image object from the modified array
out = Image.fromarray(arr)

# Save the image with a transparent background
out.save(path + "00011A02_transparent.png")# import the numpy package
