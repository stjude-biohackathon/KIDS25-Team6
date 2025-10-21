from fastapi import FastAPI,Query,status,HTTPException,Depends,File,UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

from ML_model import prediction
from fastapi.responses import JSONResponse
import ast

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_prediction_results(uploaded_files, checkpoint_path='checkpoints/molnextr_best.pth'):
    return prediction.predict_from_image_files(uploaded_files, checkpoint_path)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/prediction-only",status_code=status.HTTP_200_OK)
async def run_prediction(model: str = Form(...), images: str = Form(...)):
    # call the agent to classify the image

    


    # now we need to call the predition function from ML_model
    images_list = [os.path.join("decimer-api", img) for img in ast.literal_eval(images)]
    checkpoint_path = os.path.join("ML_Model/checkpoints", model)
    
    start_time = datetime.now()
    results = await get_prediction_results(images_list, checkpoint_path=checkpoint_path)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    results = await get_prediction_results(images_list, checkpoint_path=checkpoint_path)
    
    results_with_files = [
        {
            "filename": os.path.basename(image_path),
            "filepath": image_path,
            "processing_time": processing_time,
            **res
        }
        for image_path, res in zip(ast.literal_eval(images), results)
    ]
    return JSONResponse(content={
        "results": results_with_files
    })
    # return JSONResponse(content={
    #     # "file_paths": images_list,
    #     # "message": "Files uploaded successfully" + checkpoint_path,
    #     "results": results
    # })

@app.post("/prediction",status_code=status.HTTP_200_OK)
async def upload_and_run_prediction(file: UploadFile = File(...)):
    uploaded_file = []
    try:
        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename, ext = os.path.splitext(file.filename)
        new_filename = f"{filename}_{timestamp}{ext}"
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, new_filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        uploaded_file.append(file_path)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()
    
    # now we need to call the predition function from ML_model
    results = await get_prediction_results(uploaded_file, checkpoint_path='checkpoints/molnextr_best.pth')
    return JSONResponse(content={
        "filename": new_filename,
        "file_paths": uploaded_file,
        "message": "File uploaded successfully",
        "results": results
    })
    
@app.post("/predictions",status_code=status.HTTP_200_OK)
async def upload_and_run_predictions(files: list[UploadFile]=File(...)):
    uploaded_files = []
    for file in files:
        try:
            contents = await file.read()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename, ext = os.path.splitext(file.filename)
            new_filename = f"{filename}_{timestamp}{ext}"
            upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, new_filename)
            with open(file_path, "wb") as f:
                f.write(contents)
            uploaded_files.append(file_path)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            await file.close()

    # now we need to call the predition function from ML_model
    results = await get_prediction_results(uploaded_files, checkpoint_path='checkpoints/molnextr_best.pth')
    return JSONResponse(content={
        "file_paths": uploaded_files,
        "message": "Files uploaded successfully",
        "results": results
    })

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    
    return {"item_id": item_id, "q": q}

# @app.post("/uploadfile",status_code=status.HTTP_200_OK)
# async def create_upload_file(file: UploadFile=File(...)):
#     return {"filename": file.filename,"filesize":len(file.file.read())}

# for Individual files
@app.post("/uploadfile", status_code=status.HTTP_200_OK)
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename, ext = os.path.splitext(file.filename)
        new_filename = f"{filename}_{timestamp}{ext}"
        os.makedirs("uploads", exist_ok=True)
        with open(f"uploads/{new_filename}", "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()
    return {"filename": new_filename, "message": "File uploaded successfully"}

# for multiple files
@app.post("/uploadfiles",status_code=status.HTTP_200_OK)
async def create_upload_files(filess: list[UploadFile]=File(...)):
    return {"filename": [file.filename for file in filess],"filesize":[len(file.file.read()) for file in filess]}