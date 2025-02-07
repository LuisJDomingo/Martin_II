# Martin_II

# Proyecto de Clasificación de Sentimientos con RNN

Este proyecto implementa un modelo de red neuronal recurrente (RNN) para la clasificación de sentimientos en reseñas de películas utilizando el conjunto de datos IMDB.

## Descripción

El objetivo de este proyecto es entrenar un modelo de RNN para clasificar reseñas de películas como positivas o negativas. Utiliza la biblioteca `torchtext` para el procesamiento de texto y `PyTorch` para la construcción y entrenamiento del modelo.

## Requisitos

- Python 3.6 o superior
- PyTorch
- torchtext
- tqdm
- spacy
- Conjunto de datos IMDB

## Instalación

1. Clona el repositorio:
```sh
   git clone https://github.com/tu_usuario/Martin_II.git
   cd Martin_II
```


2. Crea y activa un entorno virtual:

```sh
    python -m venv env
    .\env\Scripts\activate  # En Windows
    source env/bin/activate  # En macOS/Linux
```
3. Instala las dependencias:

```sh
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
```
## Uso  

1. Ejecuta el script principal para entrenar el modelo:

```sh
    python MII_draft.py
```
2. El progreso del entrenamiento se mostrará en la consola, incluyendo la pérdida por lote y cualquier offset inválido encontrado.

## Estructura del Proyecto

- `MII_draft.py`: Contiene el código principal para la definición del modelo, procesamiento de datos y entrenamiento.
- `requirements.txt`: Lista de dependencias necesarias para el proyecto.

## Código Principal

**Definición del Model**

```python
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
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            return self.fc(hidden.squeeze(0))
```
**Función `yield_tokens`**

```python
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
```

**Función `train_model`**

```python
    def train_model():
        def collate_batch(batch):
            label_list, text_list, offsets = [], [], []
            current_offset = 0
            for (_label, _text) in batch:
                label_list.append(int(_label == 'pos'))
                processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
                text_list.append(processed_text)
                offsets.append(current_offset)
                current_offset += processed_text.size(0)
            label_list = torch.tensor(label_list, dtype=torch.int64)
            offsets = torch.tensor(offsets, dtype=torch.int64)
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

        for epoch in range(5):
            print(f"Epoch {epoch+1}")
            for i, (label, text, offsets) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
                optimizer.zero_grad()
                if torch.any(offsets <= 0):
                    print(f"Invalid offsets found: {offsets}")
                    continue
                predictions = model(text, offsets)
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(1)
                predictions = predictions.view(-1)
                loss = criterion(predictions, label.float())
                loss.backward()
                optimizer.step()
                print(f"Batch {i+1} Loss: {loss.item()}")
                
```

**Contribuciones**
Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para cualquier mejora o corrección.

**Licencia**
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
