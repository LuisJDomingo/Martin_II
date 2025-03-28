import speech_recognition as sr

# Inicializa el recognizer
recognizer = sr.Recognizer()

# Usa un archivo de audio como entrada (por ejemplo, un archivo WAV o MP3)
audio_file = "audio.wav"  # Cambia este nombre al archivo que quieras transcribir

# Abre el archivo de audio
with sr.AudioFile(audio_file) as source:
    print("Escuchando el audio...")
    audio_data = recognizer.record(source)  # Lee todo el archivo de audio

# Intenta reconocer el texto desde el audio
try:
    print("Transcribiendo audio...")
    text = recognizer.recognize_google(audio_data, language="es-ES")  # Usa el API de Google para la transcripción
    print("Texto transcrito:")
    print(text)
except sr.UnknownValueError:
    print("No se pudo entender el audio")
except sr.RequestError as e:
    print(f"Error al contactar con el servicio de Google: {e}")
