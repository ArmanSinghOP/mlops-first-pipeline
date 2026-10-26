from pathlib import Path
import joblib
# pyright: ignore [reportMissingImports]
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/iris_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Classifier API", version="1.0")

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

@app.get("/")
def welcome():
    return {"message": "Welcome to Iris Classifier API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])
    return {
        "prediction": prediction,
        "class_name": ["setosa","versicolor","virginica"][prediction]
    }
