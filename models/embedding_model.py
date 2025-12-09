import time
from .base import BaseModel

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




class EmbeddingModel(BaseModel):
    name = "sentence-transformer"
    model = None
    tokenizer = None

    def __init__(self, delay: float = 0.1):
        super().__init__()
        self.delay = delay

    def load(self) -> None:
        print('Loading Embedding Model')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self._is_loaded = True
        print('Embedding Model is loaded')

    def predict(self, prompt: str) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        # Return a simple echoed response for demo/testing
        return f"[sentence transformer prediction] {prompt}"


    def get_embeddings(self, sentence: str) -> list:
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings[0].tolist()