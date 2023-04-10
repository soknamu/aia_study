import os
import numpy as np
from PIL import Image
from rembg import remove
import time

for i in range(53,81):
    input_dir = f"d:study_data/asian_data/{i}/"
    output_dir = f"d:study_data/_data/asian_data/{i}/"
    print(f'{i}번 폴더 시작')
    start_time=time.time()
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all the files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Remove the background
            out = remove(img)

            # Convert the output image to RGBA mode
            out = out.convert('RGBA')

            # Create a NumPy array from the image
            arr = np.array(out)

            # Set the alpha channel to 0 for pixels below a threshold value (in this case, 100)
            arr[(arr[:,:,0] < 1) & (arr[:,:,1] < 1) & (arr[:,:,2] < 1)] = [0, 0, 0, 0]

            # Create a new Image object from the modified array
            out = Image.fromarray(arr)

            # Save the image with a transparent background
            output_path = os.path.join(output_dir, filename.split(".")[0] + "_transparent.png")
            out.save(output_path)
    print(f'{i}번 폴더 끝')
    print(f'runtime : {time.time()-start_time}')
