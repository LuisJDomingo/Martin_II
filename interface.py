import gradio as gr

# Función que usa tu modelo LLM
def chat_model(prompt):
    respuesta = f"Procesando: {prompt}"  # Aquí deberías conectar tu modelo LLM real
    return respuesta

# Crear la interfaz
iface = gr.Interface(
    fn=chat_model, 
    inputs="text", 
    outputs="text",
    title="Chat con LLM",
    description="Escribe un mensaje y el modelo responderá."
)

# Ejecutar la interfaz
iface.launch()
