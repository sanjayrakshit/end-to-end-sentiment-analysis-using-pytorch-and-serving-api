import torch
from transformers import BertTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from dataset_preparation import clean_and_lemmatize
from models import LSTMModel
import spacy

tokenizer = BertTokenizer.from_pretrained('./bert-tokenizer')
lstm_model = LSTMModel(vocab_size=tokenizer.vocab_size,
                       embed_size=50,
                       lstm_out=50,
                       lstm_layers=2,
                       out_dim=1,
                       dropout_lstm=0.6,
                       dropout=0.6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model.to(device)
loaded_chekpoints = torch.load('models/model-55.pt', map_location=device)
lstm_model.load_state_dict(loaded_chekpoints['model_state_dict'])
lstm_model.eval()

nlp = spacy.load('en_core_web_sm')

app = FastAPI()


class Review(BaseModel):
    text: str


@app.post('/predict')
def predict_sentiment(payload: Review):
    cleaned_text = clean_and_lemmatize(payload.text)
    tokens = tokenizer.tokenize(cleaned_text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    length = len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).view(1, -1).to(device)
    length = [length]
    out, = lstm_model(input_ids=input_ids, input_lengths=length)
    score = out.item()
    tag = None
    if score <= 0.25:
        tag = 'Worst'
    elif score <= 0.50:
        tag = 'Bad'
    elif score <= 0.75:
        tag = 'Good'
    else:
        tag = 'Great'
    return dict(review=payload.text, score=score, tag=tag)
