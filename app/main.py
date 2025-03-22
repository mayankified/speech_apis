from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="Speech Processing API")

# Include API routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    # Load models on startup
    from app.models import emotion_model, lung_model
    emotion_model.load_model()  # Load emotion detection model
    lung_model.load_model()     # Load lung disease detection model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
