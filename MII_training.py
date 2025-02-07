import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# Tokenizador y vocabulario
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
print("Tokenizer loaded")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = IMDB(split=('train', 'test'))
print("Data iterators created")

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
print("Vocabulary built")

vocab.set_default_index(vocab["<unk>"])
print("Default index set for vocabulary")

# Configuraciones y entrenamiento
def train_model():

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]  # Comienza en 0
        
        for _label, _text in batch:
            label_list.append(int(_label == 'pos'))
            processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(offsets[-1] + processed_text.size(0))  # Acumula el tamaño del texto

        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1])  # Se excluye el último valor para mantener la estructura correcta
        text_list = torch.cat(text_list)

        return label_list, text_list, offsets


    train_iter, test_iter = IMDB(split=('train', 'test'))
    train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)

    input_dim = len(vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1

    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(torch.device('cpu'))
    criterion = criterion.to(torch.device('cpu'))
    invalid_offsets = 0
    total_offsets = 0
    for epoch in range(5):
        print(f"Epoch {epoch+1}")
        total_offsets += 1
        for i, (label, text, offsets) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            optimizer.zero_grad()

            if torch.any(offsets <= 0):
                #print(f"Invalid offsets found: {offsets}")
                invalid_offsets += 1 
                continue

            predictions = model(text, offsets)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(1)
            predictions = predictions.view(-1)
            loss = criterion(predictions, label.float())
            loss.backward()
            optimizer.step()
            #print(f"Batch {i+1} Loss: {loss.item()}")
    print("invalid_offsets", invalid_offsets)
    print("total_offsets", total_offsets)
# Llamada a la función train_model
if __name__ == "__main__":
    train_model()