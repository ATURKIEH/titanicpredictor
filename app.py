from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from predict import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    features: dict


@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/predict")
def make_prediction(item: Item):
    results = predict(item.features)
    return results



