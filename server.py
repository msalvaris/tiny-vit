from fastapi import FastAPI, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import StreamingResponse
import os
import io
from PIL import Image
import re

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# @app.get("/")
# async def read_root(request: Request):
#     # List all files in the image directory
#     image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]
#     return templates.TemplateResponse("index.html", {"request": request, "images": image_files})




# ... [the rest of your imports]

def create_thumbnails(image_files, directory):
    print("creating thumbnails")
    base_height = 600
    os.makedirs(os.path.join(directory, "thumbnails"), exist_ok=True)
    for img_file in image_files:
         # Open an image with PIL
        img = Image.open(os.path.join(directory, img_file))
        # directory,fname = os.path.split(img_file)

        # Resize while maintaining the aspect ratio
        hpercent = base_height / float(img.size[1])
        wsize = int(float(img.size[0]) * float(hpercent))
        img = img.resize((wsize, base_height), Image.LANCZOS)

        # Save image to buffer
        img.save(os.path.join(directory, "thumbnails", img_file), format='JPEG', quality=85)

    return os.path.join(directory, "thumbnails")

def extract_numbers(filename):
    # Extract x and y values from the filename
    match = re.search(r'Epoch_(\d+)_(\d+).png', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

@app.get("/")
async def read_root(request: Request, images_dir: str = "images"):
    print(images_dir)
    print(os.getcwd())
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    # List all files in the image directory
    image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))], 
                         key=extract_numbers)
    print(image_files)
    thumbnail_dir_name = create_thumbnails(image_files, images_dir)
    image_files = sorted([f for f in os.listdir(thumbnail_dir_name) if os.path.isfile(os.path.join(thumbnail_dir_name, f))], 
                         key=extract_numbers)
    print(image_files)
    return templates.TemplateResponse("index.html", {"request": request, "images": image_files, "images_dir": thumbnail_dir_name})


@app.get("/images/{path:path}")
async def serve_image(path: str):
    file_path = path
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
          
    with open(file_path, "rb") as f:
        content = f.read()
    return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # For images in images directory: http://127.0.0.1:8000/?images_dir=images
    # For another directory other_folder: http://127.0.0.1:8000/?images_dir=other_folder

