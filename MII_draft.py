import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader

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
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            print(f"Label: {_label}")
            print(f"Text: {_text}")
            label_list.append(int(_label == 'pos'))
            processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
            print(f"Processed text: {processed_text}")
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        print(f"Batch collated: labels={label_list.size()}, text={text_list.size()}, offsets={offsets.size()}")
        print(f"Offsets: {offsets}")  # Agrega esta línea para imprimir los offsets
        return label_list, text_list, offsets

    train_iter, test_iter = IMDB(split=('train', 'test'))
    print("Train and test iterators created")

    train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
    print("Train dataloader created")
    print(len(train_dataloader), train_dataloader)

    test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
    print("Test dataloader created")

    input_dim = len(vocab)
    print(f"Input dimension: {input_dim}")

    embedding_dim = 100
    print(f"Embedding dimension: {embedding_dim}")

    hidden_dim = 256
    print(f"Hidden dimension: {hidden_dim}")

    output_dim = 1
    print(f"Output dimension: {output_dim}")

    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
    print("Model created")

    optimizer = optim.Adam(model.parameters())
    print("Optimizer created")

    criterion = nn.BCEWithLogitsLoss()
    print("Criterion created")

    model = model.to(torch.device('cpu'))
    print("Model moved to CPU")

    criterion = criterion.to(torch.device('cpu'))
    print("Criterion moved to CPU")

    for epoch in range(5):
        print(f"Epoch {epoch+1}")
        for i, (label, text, offsets) in enumerate(train_dataloader):
            print(f"Batch {i+1}")
            print(f"Label: {label}")
            print(f"Text: {text}")
            print(f"Offsets: {offsets}")

            optimizer.zero_grad()
            print("Optimizer gradients zeroed")

            # Verificar que todos los offsets sean mayores que cero
            if torch.any(offsets <= 0):
                print(f"Invalid offsets found: {offsets}")
                continue

            predictions = model(text, offsets)
            print(f"Predictions: {predictions}")

            if predictions.dim() > 1:
                predictions = predictions.squeeze(1)  # Asegúrate de que las dimensiones coincidan
            print(f"Predictions after squeeze: {predictions}")

            predictions = predictions.view(-1)  # Asegúrate de que las dimensiones coincidan
            print(f"Predictions after view: {predictions}")

            loss = criterion(predictions, label.float())
            print(f"Loss: {loss.item()}")

            loss.backward()
            print("Loss backward pass completed")

            optimizer.step()
            print("Optimizer step completed")

if __name__ == "__main__":
    train_model()