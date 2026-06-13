from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("wine_model.pkl")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
async def predict(data: dict):

    features = [[
        data["alcohol"],
        data["malic_acid"],
        data["ash"],
        data["alcalinity_of_ash"],
        data["magnesium"],
        data["total_phenols"],
        data["flavanoids"],
        data["nonflavanoid_phenols"],
        data["proanthocyanins"],
        data["color_intensity"],
        data["hue"],
        data["od280_od315"],
        data["proline"]
    ]]

    prediction = model.predict(features)

    return JSONResponse({
        "prediction": int(prediction[0])
    })