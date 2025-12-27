from fastapi import FastAPI
from api.health.router import router as health_router
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("src/templates/index.html").read_text(encoding="utf-8")


app.include_router(health_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 74))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=True)
