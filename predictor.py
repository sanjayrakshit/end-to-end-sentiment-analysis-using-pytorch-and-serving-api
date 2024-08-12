import torch
from transformers import BertTokenizer

from dataset_preparation import clean_and_lemmatize
from models import LSTMModel


def predict(text, model, tokenizer, device):
    """
    Predictor function which takes care of preprocessing and prediction
    :param text:
    :param model:
    :param tokenizer:
    :param device:
    :return:
    """
    cleaned_text = clean_and_lemmatize(text)
    tokens = tokenizer.tokenize(cleaned_text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    length = len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).view(1, -1).to(device)
    length = [length]
    out, = model(input_ids=input_ids, input_lengths=length)
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
    return dict(review=text, score=score, tag=tag)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-tokenizer')

    loaded_chekpoints = torch.load('models/model-55.pt', map_location=device)
    model = LSTMModel(vocab_size=tokenizer.vocab_size,
                      embed_size=50,
                      lstm_out=50,
                      lstm_layers=2,
                      out_dim=1,
                      dropout_lstm=0.6,
                      dropout=0.6)
    model.load_state_dict(loaded_chekpoints['model_state_dict'])
    model.to(device)
    model.eval()
    while True:
        text = input("Enter review: ")
        text = text.strip()
        r = predict(text, model, tokenizer, device)
        print(r)
