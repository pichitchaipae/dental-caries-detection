import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch

from services.validator import PanoramicValidator
from services.dental_pipeline import DentalPipelineOrchestrator
from services.renderer import render_predictions
import uuid

app = FastAPI(title="Medical AI - Caries Detection Pipeline")

# Mount outputs directory so frontend can access the annotated images
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

TEMP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

validator = PanoramicValidator()
pipeline = DentalPipelineOrchestrator()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Verify Content-Type
    if image.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "Invalid file type. Only image/png and image/jpeg are supported."
        })
        
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, image.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # ---------------------------------------------------------
        # Resource Tracking: Start
        # ---------------------------------------------------------
        start_time = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
            
        # ---------------------------------------------------------
        # STAGE 1: Panoramic Validation Model
        # ---------------------------------------------------------
        try:
            val_result = validator.validate(temp_file_path)
        except FileNotFoundError as e:
            return JSONResponse(status_code=500, content={
                "success": False,
                "error": str(e)
            })
            
        if not val_result.get("success", False):
            return JSONResponse(status_code=422, content=val_result)
            
        # ---------------------------------------------------------
        # STAGE 2: Dental Pipeline Processing
        # ---------------------------------------------------------
        request_id = str(uuid.uuid4())
        inference_response = pipeline.process_dental_radiograph(temp_file_path, request_id)
            
        # ---------------------------------------------------------
        # STAGE 3: Bounding Box Rendering
        # ---------------------------------------------------------
        # Caries findings (red boxes with surface label)
        render_preds = []
        for finding in inference_response.findings:
            render_preds.append({
                "tooth": finding.fdi,
                "surface": finding.surface,
                "confidence": finding.rf_confidence,
                "bbox": finding.bbox_xyxy,
            })

        # All detected teeth (for grey boxes on healthy teeth)
        all_tooth_bboxes = inference_response.all_tooth_bboxes or []

        annotated_image_rel_path = render_predictions(
            temp_file_path, render_preds, OUTPUTS_DIR,
            all_tooth_bboxes=all_tooth_bboxes,
        )

        # Clean up temp file
        os.remove(temp_file_path)

        # ---------------------------------------------------------
        # API Response
        # ---------------------------------------------------------
        response_dict = inference_response.model_dump()
        response_dict["validation"] = val_result.get("validation", {})
        response_dict["processedImage"] = f"/{annotated_image_rel_path}"

        return response_dict
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

@app.get("/")
def read_root():
    return {"message": "FastAPI Medical AI Inference Backend Operational."}
