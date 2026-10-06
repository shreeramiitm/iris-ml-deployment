import os
import sys
import logging
import pickle
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Set up standard Python logging directed to stdout at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("iris_api")

# Global variables for model state
model: Any = None
model_loaded: bool = False

# Class to species mapping: 0 -> setosa, 1 -> versicolor, 2 -> virginica
SPECIES_MAPPING: Dict[int, str] = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager to safely load model.pkl on startup.
    """
    global model, model_loaded
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl"),
        os.path.join(os.getcwd(), "model", "model.pkl"),
        "model/model.pkl"
    ]

    model_path = None
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            model_path = abs_p
            break

    if model_path:
        logger.info(f"Loading model artifact from: {model_path}")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_loaded = True
            logger.info("Model loaded successfully into memory.")
        except Exception as e:
            logger.error(f"Failed to load model file from {model_path}: {e}", exc_info=True)
            model_loaded = False
    else:
        logger.error(f"Model file not found in expected paths: {possible_paths}")
        model_loaded = False

    yield

    logger.info("Shutting down Iris Classification API.")


# Initialize FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="Production-ready FastAPI service for Iris species classification using KNeighborsClassifier.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if it exists
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    """
    Serve static index.html single-page frontend application.
    """
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Iris Classification API is running. Access /docs for API documentation."}


class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", json_schema_extra={"example": 5.1})
    sepal_width: float = Field(..., description="Sepal width in cm", json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., description="Petal length in cm", json_schema_extra={"example": 1.4})
    petal_width: float = Field(..., description="Petal width in cm", json_schema_extra={"example": 0.2})

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="Predicted target class integer (0, 1, or 2)")
    species: str = Field(..., description="Mapped species name ('setosa', 'versicolor', or 'virginica')")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """
    Health check endpoint returning application status and model load state.
    """
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(payload: IrisInput):
    """
    Classify Iris flower species based on input feature measurements.
    """
    logger.info(
        f"Incoming request -> sepal_length={payload.sepal_length}, "
        f"sepal_width={payload.sepal_width}, petal_length={payload.petal_length}, "
        f"petal_width={payload.petal_width}"
    )

    if not model_loaded or model is None:
        logger.error("Prediction failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    try:
        # Construct 2D numpy array in exact required feature order:
        # [sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)]
        features = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width
        ]], dtype=np.float64)

        prediction_array = model.predict(features)
        predicted_class = int(prediction_array[0])
        species = SPECIES_MAPPING.get(predicted_class, "unknown")

        logger.info(f"Prediction result -> predicted_class={predicted_class}, species='{species}'")

        return PredictionResponse(
            predicted_class=predicted_class,
            species=species
        )
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction processing error: {str(e)}"
        )
