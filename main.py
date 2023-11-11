from fastapi import FastAPI, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import subprocess
from pydantic import BaseModel
from fastapi.responses import HTMLResponse


app = FastAPI()
UPLOAD_DIR = "images"

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "output"

@app.get("/list-images/", response_class=HTMLResponse)
async def list_images():
    images = []
    for filename in os.listdir(UPLOADS_DIR):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(UPLOADS_DIR, filename)
            images.append(image_path)

    return HTMLResponse(content=f"<html><body>{'<br>'.join(images)}</body></html>")

def process_file(file_path, output_folder):
    try:
        # Pass the file path as an argument to the solution.py script
        result = subprocess.run(["python", "solution.py", file_path], capture_output=True, text=True, check=True)
        execution_output = result.stdout

        # Return URL to access the saved image
        output_image_url = f"{output_folder}/{os.path.basename(file_path) + '2'}"
    except subprocess.CalledProcessError as e:
        execution_output = str(e)
        output_image_url = None

    return execution_output, output_image_url


OUTPUT_DIR = "output"
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), is_video: bool = False):
    try:
        print(f"Received file upload request: {file.filename}")

        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if is_video:
            output_folder = "output_with_weapon" if "weapon" in file.filename else "output"
            execution_output, output_image_url = process_file(file_path, output_folder)
        else:
            execution_output, output_image_url = process_file(file_path, OUTPUT_DIR)

        if os.path.exists(file_path):
            os.remove(file_path)  # Remove the uploaded file

        if is_video:
            return JSONResponse(content={"message": "Файл успешно загружен", "execution_output": execution_output, "output_image_url": output_image_url})
        else:
            image_url = f"/images/{file.filename}"
            return JSONResponse(content={"message": "Файл успешно загружен", "image_url": image_url, "execution_output": execution_output, "output_image_url": output_image_url})
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"message": "Ошибка при загрузке файла"}, status_code=500)



@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(UPLOAD_DIR, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        return JSONResponse(content={"message": "Изображение не найдено"}, status_code=404)

@app.get("/output/{image_name}")
async def get_output_image(image_name: str):
    # Определите путь к изображению в папке "output"
    image_path = os.path.join("output", image_name)
    if os.path.exists(image_path):
        # Если изображение существует, верните его как файловый ответ
        return FileResponse(image_path)
    else:
        return JSONResponse(content={"message": "Изображение не найдено в папке 'output'"}, status_code=404)


class VideoURLRequest(BaseModel):
    videoURL: str

@app.post("/upload-video/")
async def upload_video(request_data: VideoURLRequest):
    video_url = request_data.videoURL


    print("Received video URL:", video_url)
    return JSONResponse(content={"message": "Ссылка на видео успешно получена", "videoURL": video_url})


@app.post("/send-video/")
async def send_video(request_data: VideoURLRequest):
    video_url = request_data.videoURL
    return JSONResponse(content={"message": "Ссылка на видео успешно получена", "videoURL": video_url})


@app.get("/output/")
async def list_output_images():
    images = []
    output_folder_path = "output"

    # List all files in the "output" folder
    for filename in os.listdir(output_folder_path):
        if filename.endswith(".jpg"):
            image_url = f"/output/{filename}"
            images.append(image_url)

    return JSONResponse(content={"images": images})
