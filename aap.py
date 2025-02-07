from flask import Flask, request, jsonify
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
import torch.nn as nn

app = Flask(__name__)

# Definir el modelo
class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, offsets):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, offsets, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(hidden.squeeze(0))

# Tokenizer y vocabulario
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, _ = IMDB(split=('train', 'test'))
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Cargar el modelo entrenado
input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Preprocesar el texto
    processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64).unsqueeze(0)
    offsets = torch.tensor([0], dtype=torch.int64)
    
    # Hacer la predicción
    with torch.no_grad():
        prediction = model(processed_text, offsets)
        prediction = torch.sigmoid(prediction).item()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)