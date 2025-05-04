import gradio as gr
#import transformers
from transformers import pipeline

# Cargar un modelo LLM preentrenado (por ejemplo, GPT-2)
llm = pipeline("text-generation", model="gpt2")

# Función para manejar entradas de texto
def generate_text(prompt, max_length=50):
    response = llm(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]

# Función para manejar entradas de audio (transcripción)
def transcribe_audio(audio):
    import speech_recognition as sr
    import os

    # Verificar si el archivo existe
    if not os.path.isfile(audio):
        return "Error: No se pudo encontrar el archivo de audio."

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"Error al procesar el audio: {str(e)}"

# Función para manejar entradas de imagen (descripción)
def describe_image(image):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Configurar la interfaz con múltiples entradas y salidas
with gr.Blocks() as demo:
    gr.Markdown("# Interfaz con soporte para múltiples entradas y salidas")
    
    with gr.Tab("Texto"):
        text_input = gr.Textbox(label="Escribe un texto")
        text_output = gr.Textbox(label="Respuesta del modelo")
        text_button = gr.Button("Generar")
        text_button.click(generate_text, inputs=[text_input], outputs=[text_output])
    
    with gr.Tab("Audio"):
        audio_input = gr.Audio(type="filepath", label="Sube un archivo de audio")  # Ajustado
        audio_output = gr.Textbox(label="Transcripción")
        audio_button = gr.Button("Transcribir")
        audio_button.click(transcribe_audio, inputs=[audio_input], outputs=[audio_output])
    
    with gr.Tab("Imagen"):
        image_input = gr.Image(type="filepath", label="Sube una imagen")
        image_output = gr.Textbox(label="Descripción generada")
        image_button = gr.Button("Describir")
        image_button.click(describe_image, inputs=[image_input], outputs=[image_output])

# Ejecutar la interfaz
if __name__ == "__main__":
    demo.launch(share=True)
