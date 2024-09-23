import gradio as gr  # Importa la biblioteca Gradio para crear interfaces web interactivas.
import whisper  # Importa la biblioteca Whisper para transcripción de audio.
from translate import Translator  # Importa la clase Translator para traducción de texto.
from dotenv import dotenv_values  # Importa dotenv_values para cargar variables de entorno desde un archivo .env.
from elevenlabs.client import ElevenLabs  # Importa la clase ElevenLabs para convertir texto a voz.
from elevenlabs import VoiceSettings  # Importa VoiceSettings para configurar la voz en ElevenLabs.

# Carga las variables de entorno desde el archivo .env
config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]  # Obtiene la clave API de ElevenLabs desde las variables de entorno.

# Función para dividir el texto en partes más pequeñas
def chunk_text(text, max_len=500):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

# Función principal del traductor
def translator(audio_file):

    # 1. Transcribir texto

    try:
        model = whisper.load_model("base")  # Carga el modelo base de Whisper.
        result = model.transcribe(audio_file, language="Spanish", fp16=False)  # Transcribe el archivo de audio en español.
        transcription = result["text"]  # Obtiene el texto transcrito.
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")  # Maneja errores en la transcripción.

    print(f"Texto original: {transcription}")  # Imprime el texto transcrito original.

    # Divide la transcripción en partes más pequeñas si es muy larga
    transcription_chunks = chunk_text(transcription)

    # 2. Traducir texto
    translations = {
        "en": "",  # Traducción al inglés.
        "it": "",  # Traducción al italiano.
        "fr": "",  # Traducción al francés.
        "ja": ""   # Traducción al japonés.
    }

    try:
        for chunk in transcription_chunks:
            translations["en"] += Translator(from_lang="es", to_lang="en").translate(chunk) + " "  # Traduce al inglés.
            translations["it"] += Translator(from_lang="es", to_lang="it").translate(chunk) + " "  # Traduce al italiano.
            translations["fr"] += Translator(from_lang="es", to_lang="fr").translate(chunk) + " "  # Traduce al francés.
            translations["ja"] += Translator(from_lang="es", to_lang="ja").translate(chunk) + " "  # Traduce al japonés.
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traduciendo el texto: {str(e)}")  # Maneja errores en la traducción.

    print(f"Texto traducido a Inglés: {translations['en']}")  # Imprime la traducción al inglés.
    print(f"Texto traducido a Italiano: {translations['it']}")  # Imprime la traducción al italiano.
    print(f"Texto traducido a Francés: {translations['fr']}")  # Imprime la traducción al francés.
    print(f"Texto traducido a Japonés: {translations['ja']}")  # Imprime la traducción al japonés.

    # 3. Generar audio traducido

    en_save_file_path = text_to_speach(translations["en"], "en")  # Genera el audio en inglés.
    it_save_file_path = text_to_speach(translations["it"], "it")  # Genera el audio en italiano.
    fr_save_file_path = text_to_speach(translations["fr"], "fr")  # Genera el audio en francés.
    ja_save_file_path = text_to_speach(translations["ja"], "ja")  # Genera el audio en japonés.

    return en_save_file_path, it_save_file_path, fr_save_file_path, ja_save_file_path  # Devuelve las rutas de los archivos de audio generados.

# Función para convertir texto a voz
def text_to_speach(text: str, language: str) -> str:

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)  # Crea un cliente de ElevenLabs con la clave API.

        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam  # ID de la voz a utilizar.
            optimize_streaming_latency="0",  # Optimiza la latencia de transmisión.
            output_format="mp3_22050_32",  # Formato de salida del audio.
            text=text,  # Texto a convertir en audio.
            model_id="eleven_turbo_v2",  # ID del modelo a utilizar.
            voice_settings=VoiceSettings(
                stability=0.0,  # Configuración de estabilidad de la voz.
                similarity_boost=0.0,  # Configuración de aumento de similitud.
                style=0.0,  # Configuración de estilo de la voz.
                use_speaker_boost=True,  # Uso de aumento de altavoz.
            ),
        )

        save_file_path = f"audios/{language}.mp3"  # Ruta donde se guardará el archivo de audio.

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)  # Escribe los datos del audio en el archivo.

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error creando el audio: {str(e)}")  # Maneja errores en la generación de audio.

    return save_file_path  # Devuelve la ruta del archivo de audio generado.

# Configuración de la interfaz web con Gradio
web = gr.Interface(
    fn=translator,  # Función a llamar cuando se use la interfaz.
    inputs=gr.Audio(
        sources=["microphone","upload"],  # Fuentes de audio permitidas.
        type="filepath",  # Tipo de entrada de audio.
        label="Español"  # Etiqueta para la entrada de audio.
    ),
    outputs=[
        gr.Audio(label="Inglés"),  # Salida de audio en inglés.
        gr.Audio(label="Italiano"),  # Salida de audio en italiano.
        gr.Audio(label="Francés"),  # Salida de audio en francés.
        gr.Audio(label="Japonés")  # Salida de audio en japonés.
    ],
    title="Traductor de voz",  # Título de la interfaz.
    description="Traductor de voz con IA a varios idiomas"  # Descripción de la interfaz.
)

web.launch()  # Lanza la interfaz web.
