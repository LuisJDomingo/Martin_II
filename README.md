# Martin II - Procesamiento de Lenguaje Natural y Redes Neuronales

Este proyecto implementa un conjunto de programas para realizar tareas de procesamiento de lenguaje natural (NLP) y aprendizaje profundo utilizando PyTorch, TorchText y Flask. El objetivo principal es entrenar y servir modelos de redes neuronales para tareas como clasificación de texto y transcripción de audio.

## Estructura del Proyecto

- **`app.py`**: Implementa una API REST con Flask para realizar predicciones utilizando un modelo RNN entrenado.
- **`audio_to_text.py`**: Convierte archivos de audio a texto utilizando la biblioteca `speech_recognition`.
- **`clean_cache.py`**: Limpia la caché de TorchText para evitar problemas con los datos almacenados localmente.
- **`interface.py`**: Proporciona una interfaz gráfica simple con Gradio para interactuar con un modelo LLM.
- **`MII_training.py`**: Entrena un modelo Transformer para clasificación de texto utilizando el dataset IMDB.
- **`Neural_network.py`**: Define una arquitectura Transformer para tareas de NLP.
- **`utils.py`**: Contiene utilidades como la exportación de datos a archivos CSV.
- **`requirements.txt`**: Lista las dependencias necesarias para ejecutar el proyecto.

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clona este repositorio:

    ```bash
        git clone https://github.com/tu_usuario/Martin_II.git
        cd Martin_II
    ```
2. Crea un entorno virtual y activa:

    ```bash
        python -m venv env
        source env/bin/activate  # En Windows: env\Scripts\activate
    ```
3. Instala las dependencias:
    ```bash
        pip install -r requirements.txt
    ```
## Uso
Entrenamiento del Modelo
Ejecuta ```MII_training.p```y para entrenar un modelo Transformer con el dataset IMDB:

```bash
        python [MII_training.py](http://_vscodecontentref_/0)
```
Servir el Modelo
Ejecuta app.py para iniciar la API REST:
    ```bash
            python [app.py](http://_vscodecontentref_/1)
    ```
La API estará disponible en http://127.0.0.1:5000/predict. Envía un POST request con un JSON como:
    ```bash
        {
            "text": "Este es un ejemplo de texto."
        }
    ```
    
