from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import shutil
from pathlib import Path
import uuid
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from utils.predictor import load_predictor
from utils.pdf_extractor import PDFExtractor
from utils.chatbot import MedicalChatbot

# Initialize FastAPI app
app = FastAPI(
    title="Medical Imaging Diagnostic System",
    description="AI-powered medical image analysis and diagnosis assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
predictors = {}
pdf_extractor = PDFExtractor()
chatbot = MedicalChatbot()

# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    image_type: str
    predicted_class: str
    confidence: float
    all_probabilities: dict
    image_path: str

class ChatRequest(BaseModel):
    message: str
    diagnosis_context: Optional[dict] = None
    pdf_context: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    response: str
    conversation_history: List[dict]


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global predictors
    
    print("Loading models...")
    try:
        # Try to load all models
        for dataset_type in ['aptos', 'ham10000', 'mura']:
            try:
                predictors[dataset_type] = load_predictor(dataset_type)
                print(f"✓ Loaded {dataset_type} model")
            except Exception as e:
                print(f"✗ Could not load {dataset_type} model: {e}")
                print(f"  Please train the model first using: python backend/train.py --dataset {dataset_type}")
        
        if not predictors:
            print("\n⚠ Warning: No models loaded. Please train models before using the API.")
    except Exception as e:
        print(f"Error during startup: {e}")


@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Medical Imaging Diagnostic System API",
        "status": "online",
        "loaded_models": list(predictors.keys())
    }


@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "available_models": list(predictors.keys()),
        "model_info": {
            "aptos": "Diabetic Retinopathy Detection (Fundus Images)",
            "ham10000": "Skin Lesion Classification (Dermoscopy)",
            "mura": "Bone Fracture Detection (X-Ray)"
        }
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    image_type: str = Form(...)
):
    """
    Predict disease from medical image
    
    Args:
        file: Medical image file
        image_type: Type of image ('aptos', 'ham10000', or 'mura')
    """
    if image_type not in predictors:
        raise HTTPException(
            status_code=400,
            detail=f"Model for {image_type} not available. Available models: {list(predictors.keys())}"
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        predictor = predictors[image_type]
        result = predictor.predict(file_path)
        
        return PredictionResponse(
            success=True,
            image_type=image_type,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            image_path=str(file_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/pdf/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from PDF medical report"""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded PDF
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}.pdf"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract information
        sections = pdf_extractor.extract_structured_info(file_path)
        key_points = pdf_extractor.summarize_key_points(file_path)
        is_medical = pdf_extractor.is_medical_report(file_path)
        
        return {
            "success": True,
            "is_medical_report": is_medical,
            "sections": sections,
            "key_points": key_points
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")
    finally:
        # Clean up PDF file
        if file_path.exists():
            file_path.unlink()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with AI about diagnosis and medical reports
    
    Args:
        message: User's message
        diagnosis_context: Optional diagnosis results
        pdf_context: Optional PDF extracted text
    """
    try:
        response = chatbot.chat_with_context(
            user_message=request.message,
            diagnosis_context=request.diagnosis_context,
            pdf_context=request.pdf_context
        )
        
        return ChatResponse(
            success=True,
            response=response,
            conversation_history=chatbot.get_conversation_history()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/chat/clear")
async def clear_chat_history():
    """Clear chat conversation history"""
    chatbot.clear_history()
    return {"success": True, "message": "Chat history cleared"}


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve uploaded image"""
    # Find image with any extension
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        image_path = UPLOAD_DIR / f"{image_id}{ext}"
        if image_path.exists():
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n{'='*60}")
    print("Medical Imaging Diagnostic System - Backend API")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
