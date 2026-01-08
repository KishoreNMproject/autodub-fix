from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Personalized Auto Dubbing Backend")

app.include_router(router)

@app.get("/")
def health():
    return {"status": "Backend is running"}
