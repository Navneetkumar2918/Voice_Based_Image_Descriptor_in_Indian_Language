from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from main import MultilingualImageToSpeech
from detect_and_caption import MultilingualYOLODetector

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
caption_model = MultilingualImageToSpeech()
yolo_model = MultilingualYOLODetector(model_path="best.pt")  # Update path to your YOLO model

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        contents = await file.read()
        filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
        image_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        
        with open(image_path, "wb") as f:
            f.write(contents)

        print(f"📁 Saved uploaded image: {image_path}")

        # Process with main.py - get captions with base64 audio
        print("🔄 Processing image captions...")
        results_main = caption_model.process_image_to_speech(
            image_path, 
            play_audio=False, 
            save_audio=False,
            return_base64_audio=True
        )
        
        # Process with detect_and_caption.py - get object detection
        print("🔄 Processing object detection...")
        detection_summary = yolo_model.get_detection_summary(image_path)

        # Clean up temporary file
        try:
            os.remove(image_path)
            print(f"🗑️ Cleaned up temporary file: {image_path}")
        except:
            pass

        print("✅ Analysis complete, sending response...")
        
        return JSONResponse({
            "image_caption": results_main,
            "hazard_detection": detection_summary,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        # Clean up temporary file in case of error
        try:
            if 'image_path' in locals():
                os.remove(image_path)
        except:
            pass
            
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/")
async def root():
    return {"message": "Multilingual Image Analysis API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "caption_model": True,
            "yolo_model": True
        },
        "supported_languages": caption_model.supported_languages
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Multilingual Image Analysis Server...")
    print("📋 Supported languages:", caption_model.supported_languages)
    uvicorn.run(app, host="0.0.0.0", port=8000)