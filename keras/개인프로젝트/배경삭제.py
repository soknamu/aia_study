from rembg import remove 
from PIL import Image 
path = "./asian_data/2/"

# Import the alpha_matting package
from alpha_matting import estimate_alpha

# Load the image
img = Image.open(path + "00011A02.jpg")

# Remove the background
out = remove(img)

# Estimate the alpha matte using alpha matting
alpha = estimate_alpha(img, out)

# Composite the foreground onto a new background (in this case, a solid color)
new_bg = Image.new("RGB", out.size, (255, 255, 255))
result = Image.composite(out, new_bg, alpha)

# Save the result image with a transparent background
result.save(path + "00011A02_alpha.png")
