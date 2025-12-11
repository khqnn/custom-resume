from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

from .models import initialize, register_model, get_model
from .models.dummy_model import DummyModel
from .models.embedding_model import EmbeddingModel
from .models.jdparser_model import JobDescriptionParser

app = FastAPI()


# Register models (you can add more registrations here or in separate modules)
register_model(DummyModel.name, DummyModel(delay=0.01))
register_model(EmbeddingModel.name, EmbeddingModel())
register_model(JobDescriptionParser.name, JobDescriptionParser(model_dir="./notebooks/parser/jd_tf_model"))


@app.on_event("startup")
def startup_event():
    """Initialize all registered models and load them into memory."""
    initialize()


@app.get("/predict")
def predict():
    """Run prediction using a registered model.

    Expects a JSON body: {"model": "name", "prompt": "text"}
    """
    # Look up the model by name (raises KeyError if not found)
    try:
        model_name = 'dummy'
        prompt = 'some prompt'
        model = get_model(model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Ensure the model finished loading during startup
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded")

    result = model.predict(prompt)
    return {"model": model_name, "result": result}


@app.get("/get-embeddings")
def predict():
    """Run prediction using a registered model.

    Expects a JSON body: {"model": "name", "prompt": "text"}
    """
    # Look up the model by name (raises KeyError if not found)
    try:
        sentence = 'something cool here babes...'
        model_name = 'sentence-transformer'
        model = get_model(model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Ensure the model finished loading during startup
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded")

    result = model.get_embeddings(sentence)
    return {"model": model_name, "result": result}


@app.get("/jd-parser")
def check_jd_parser():
    try:
        texts = [
            "Design APIs in Python and maintain Kubernetes deployments.",
            "Design and implement microservices in Python and maintain Kubernetes deployments.",
            "Write tests and collaborate with frontend teams."
        ]
        model_name = "job-description-parser"
        model = get_model(model_name)
    except Exception as e:
        raise e

    if not model.is_loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' not found")
    
    results = model.predict(texts)
    return {"model": model_name, "result": results}


@app.get("/")
async def root():
    return {"message": "Hello World"}