
---

### TODO.md
```markdown
# TODO - Lista de Tareas Pendientes

## General
- [ ] Crear un archivo `README.md` más detallado con ejemplos de uso y capturas de pantalla.
- [ ] Agregar pruebas unitarias para cada módulo.
- [ ] Configurar un entorno de CI/CD para pruebas automáticas.

## `app.py`
- [ ] Manejar errores en caso de que el archivo `model.pth` no exista.
- [ ] Validar la entrada JSON para evitar errores si falta el campo `text`.
- [ ] Mejorar la lógica de preprocesamiento para manejar textos más largos y offsets dinámicos.

## `audio_to_text.py`
- [ ] Agregar soporte para múltiples formatos de audio (MP3, WAV, etc.).
- [ ] Manejar errores si el archivo de audio no existe o está corrupto.
- [ ] Permitir la configuración del idioma de transcripción desde la línea de comandos.

## `clean_cache.py`
- [ ] Hacer que la ruta de la caché sea configurable.
- [ ] Agregar un mensaje de confirmación antes de eliminar la caché.

## `interface.py`
- [ ] Conectar la interfaz con un modelo LLM funcional.
- [ ] Agregar soporte para múltiples entradas y salidas (por ejemplo, imágenes o audio).

## `MII_training.py`
- [ ] Optimizar el proceso de entrenamiento para manejar grandes volúmenes de datos.
- [ ] Implementar validación durante el entrenamiento para monitorear el rendimiento del modelo.
- [ ] Guardar métricas de entrenamiento (precisión, pérdida) en un archivo o base de datos.

## `Neural_network.py`
- [ ] Documentar las clases y métodos con docstrings.
- [ ] Agregar soporte para cargar pesos preentrenados.
- [ ] Implementar pruebas para verificar la funcionalidad de la arquitectura Transformer.

## `utils.py`
- [ ] Agregar manejo de errores al exportar datos a CSV.
- [ ] Permitir la configuración de los nombres de las columnas al exportar.

## `requirements.txt`
- [ ] Verificar que todas las dependencias sean necesarias y estén actualizadas.
- [ ] Agregar instrucciones para instalar dependencias opcionales.

## Otros
- [ ] Crear un script para automatizar la configuración inicial del proyecto.
- [ ] Agregar ejemplos de uso para cada módulo en el `README.md`.