import time
from typing import List
from .base import BaseModel
from ..utils.JobDescriptionParser import load_job_description_parser_pipeline


class JobDescriptionParser(BaseModel):
    name = "job-description-parser"
    predict_method = None
    model_dir = None

    def __init__(self, model_dir=None):
        super().__init__()
        self.model_dir = model_dir        

    def load(self) -> None:
        print("Loading jd parser model")
        self.predict_method = load_job_description_parser_pipeline(self.model_dir)
        self._is_loaded = True
        print("JD parser model loaded successfully")

    def predict(self, texts: List[str]) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        # Return a simple echoed response for demo/testing
        return self.predict_method(texts)
