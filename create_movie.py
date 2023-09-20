
import imageio
import os
import glob
import re
import fire
from PIL import Image
import numpy as np


def extract_numbers(filename):
    # Extract x and y values from the filename
    match = re.search(r'Epoch_(\d+)_(\d+).png', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def resize_image(image_path, width=600):
    """Resize the image to the specified width while maintaining the aspect ratio."""
    with Image.open(image_path) as img:
        aspect_ratio = float(img.height) / float(img.width)
        height = int(aspect_ratio * width)
        img_resized = img.resize((width, height))
        return np.array(img_resized)


def create_movie(image_path:str, savename:str, fps=3):
    image_filenames = sorted(glob.glob(os.path.join(image_path, "*_3.png")), key=extract_numbers)
    
    # Read images using imageio.imread
    images = [resize_image(filename) for filename in image_filenames]

    # Write images to movie file at 3fps
    imageio.mimwrite(savename, images, fps=fps)


if __name__=="__main__":
    fire.Fire({
        "create": create_movie
    })