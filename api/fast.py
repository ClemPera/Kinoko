import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict_tab")
def predict_tab(
    cap_shape: str,
    cap_color: str,
    does_bruise_or_bleed: str,
    gill_attachment: str,
    gill_color: str,
    stem_color: str,
    has_ring: str,
    habitat: str,
    season: str
):
    pass
    # return {
    #     "fare": float(prediction)
    # }

@app.post("/predict_img")
def predict_img(
    model: str,
    image: UploadFile,
):
    
    pass
    # return {
    #     "fare": float(prediction)
    # }
