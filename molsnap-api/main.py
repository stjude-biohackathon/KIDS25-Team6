from fastapi import FastAPI,Query,status,HTTPException,Depends,File,UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime

from ML_model import prediction
from fastapi.responses import JSONResponse
import ast

# for the agent
# ref: https://lmstudio.ai/docs/app/api/endpoints/openai
# ref: https://platform.openai.com/docs/guides/images-vision?api-mode=responses#analyze-images
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
META_PROMPT = """
You are a molecule classfication agent as part of a molecule recognition pipeline. 
You will be given images of molecules from the user, and you MUST return a classification 
from the following: "macrocycle", "natural product", or "neither". 
You MUST respond with one of the classifications verbatim.
"""
# [IF USING o4-mini or some other reasoning model] 
# You must think long and hard about what type of molecule is presented. 
# If the molecule does not fit into any of the categories above, you MUST respond in one word, and that one word must be "generic".
# You MUST respond with either one word or one of the category names verbatim.
USER_PROMPT = """
Classify the given image as a macrocycle or natural product. Please return 'macrocycle' 'natural product' or 'neither'
"""

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

async def get_agent_prediction_result(uploaded_file):
    # call the agent to classify the image
    completion = client.chat.completions.create(
      model="gpt-4o", # use o1 or o4-mini if we want reasoning, will be slightly more expensive per token though.
        
      messages=[
        {"role": "system", "content": META_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Please classify the given molecule image into one of the following categories: "},
            {
                "type": "input_image",
                "image_url": uploaded_file,
            },
        ]}
      ],
      response_format={"type": "text"},
      # IF using reasoning model:
      # reasoning_effort="medium", # can be "low", "medium", or "high" for o4-mini and other similar models in the family
      
      # IF using non-reasoning model (e.g., 4o or 4.1)
      temperature=0.1, # even lower temperature is OK, but ChemEagle uses 0.1. Lower temperature = more determinism
      max_completion_tokens=100,
      top_p=1.0,
    )
    resp = completion.choices[0].message

    if resp == "macrocycle":
        # fine-tuned on macrocycles
        return prediction.predict_from_image_files([uploaded_file], 'checkpoints/molnextr_macrocycle.pth'), "macrocycle"
    elif resp == "natural product":
        # fine-tuned on natural products
        return prediction.predict_from_image_files([uploaded_file], 'checkpoints/molnextr_natprod.pth'), "natural product"
    else:
        assert resp == "neither" # hopefully the VLM follows instructions
        # default model
        return prediction.predict_from_image_files([uploaded_file], 'checkpoints/molnextr_best.pth'), ""

async def get_prediction_results(uploaded_files, checkpoint_path='checkpoints/molnextr_best.pth', use_agent=False):
    if use_agent:
        results = []
        clss = []
        for file in uploaded_files:
            res, cls = get_agent_prediction_result(file)
            results.append(res)
            clss.append(cls)

        return results, clss
    else: 
        return prediction.predict_from_image_files(uploaded_files, checkpoint_path), None

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/prediction-only",status_code=status.HTTP_200_OK)
async def run_prediction(model: str = Form(...), images: str = Form(...)):
    # now we need to call the predition function from ML_model
    images_list = [os.path.join("decimer-api", img) for img in ast.literal_eval(images)]
    checkpoint_path = os.path.join("ML_Model/checkpoints", model)
    
    start_time = datetime.now()
    results, _ = await get_prediction_results(images_list, checkpoint_path=checkpoint_path)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    results, _ = await get_prediction_results(images_list, checkpoint_path=checkpoint_path)
    
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
    results, classifications = await get_prediction_results(uploaded_file, checkpoint_path='checkpoints/molnextr_best.pth', use_agent=True)
    return JSONResponse(content={
        "filename": new_filename,
        "file_paths": uploaded_file,
        "message": f"File uploaded successfully, {classifications}", # TODO: integrate the classifications into the response message
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