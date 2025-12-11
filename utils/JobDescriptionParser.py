import os
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import joblib


def load_trained_model(model_dir):
    """
    Load a fine-tuned transformers model and tokenizer from model_dir,
    and also load a LabelEncoder saved as 'label_encoder.joblib' in the same dir.
    Returns: (tokenizer, model, label_encoder, device)
    """
    
    # 1) Check files
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # 2) Load tokenizer and model (saved by Trainer.save_model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # 3) Load label encoder - expected file name
    le_path = os.path.join(model_dir, "label_encoder.joblib")
    if not os.path.exists(le_path):
        # fallback: maybe label encoder saved elsewhere - raise informative error
        raise FileNotFoundError(f"Label encoder not found at {le_path}.")
    
    label_encoder = joblib.load(le_path)
    
    # 4) device selection
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    
    return tokenizer, model, label_encoder, device



# Inference Helper
def predict_with_model(texts, tokenizer, model, label_encoder, device=None, max_length=128, batch_size=16, top_k=1):
    """
    Predict label(s) and probabilities for a list of 'texts'.
    Returns list of dicts:
    {
        "text": orignal_text,
        "predictions": [{"label": label_str, "score": float}, ...] # length top_k sorted decs
    }
    Notes:
    - label_encoder: sklearn.preprocessing.LabelEncoder used at training time.
    - model should be the AutoModelForSequenceClassification loaded from the same model_dir.
    """

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cup")
        
    
    model.to(device)
    model.eval()

    results = []
    # batched inference
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits # shape (batch_size, num_labels)
            probs = F.softmax(logits, dim=-1).cpu().numpy() # move to cup numpy
            
            
        for j, text in enumerate(batch_texts):
            prob_row = probs[j] # shape (num_labels,)
            # get top_k indices
            topk_idx = np.argsort(prob_row)[::-1][:top_k]
            preds = []
            for idx in topk_idx:
                # map index -> label string using label_encoder
                try:
                    label_str = label_encoder.inverse_transform([int(idx)])[0]
                except Exception:
                    # fallback: if label_encoder not consistent, try to use model.config.id2label
                    id2label = getattr(model.config, "id2label", None)
                    label_str = id2label.get(str(idx), id2label.get(idx, f"LABEL_{idx}"))
                preds.append({"label": label_str, "score": float(prob_row[idx])})
                
                
            results.append({"text": text, "predictions": preds})
            
    return results


# Convenience wrapper that returns a predict function
def load_job_description_parser_pipeline(model_dir):
    """
    Load everything and return a callable predict(texts: List[str]) -> results
    """
    
    tokenizer, model, label_encoder, device = load_trained_model(model_dir)
    def predict(texts, **kwargs):
        return predict_with_model(texts, tokenizer, model, label_encoder, device=device, **kwargs)
    # attach metadata if desired
    predict.tokenizer = tokenizer
    predict.model = model
    predict.label_encoder = label_encoder
    return predict

