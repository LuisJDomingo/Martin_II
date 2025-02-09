import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import export_to_csv
from Neural_network import TransformerModel, PositionalEncoding

# Tokenizador y vocabulario
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
print("Tokenizer loaded")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = IMDB(split=('train', 'test'))
print(train_iter)
print(test_iter)
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
        print(type(label_list))
        print(type(text_list))
        print(type(offsets))
        #export_to_csv(label_list, filename='label_list.csv')
        #export_to_csv(text_list, filename='text_list.csv')
        #export_to_csv(offsets, filename='offsets.csv')
        return label_list, text_list, offsets

    train_iter, test_iter = IMDB(split=('train', 'test'))
    train_dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)

    vocab_size = len(vocab)
    d_model = 512  # Dimensión del modelo
    nhead = 8  # Número de cabezas de atención
    num_encoder_layers = 6  # Número de capas del codificador
    num_decoder_layers = 6  # Número de capas del decodificador
    dim_feedforward = 2048  # Dimensión de la capa feedforward
    max_seq_length = 100  # Longitud máxima de la secuencia
    dropout = 0.1  # Tasa de dropout

    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout)
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

            predictions = model(text, text, src_mask=None, tgt_mask=None)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(1)
            predictions = predictions.view(-1)
            loss = criterion(predictions, label.float())
            loss.backward()
            optimizer.step()
            #print(f"Batch {i+1} Loss: {loss.item()}")
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'model.pth')
    print("invalid_offsets", invalid_offsets)
    print("total_offsets", total_offsets)

# Llamada a la función train_model
if __name__ == "__main__":
    train_model()