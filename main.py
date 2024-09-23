import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

# Function to chunk text into smaller parts
def chunk_text(text, max_len=500):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def translator(audio_file):

    # 1. Transcribir texto

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo el texto: {str(e)}")

    print(f"Texto original: {transcription}")

    # Split the transcription into smaller chunks if it's too long
    transcription_chunks = chunk_text(transcription)

    # 2. Traducir texto
    translations = {
        "en": "",
        "it": "",
        "fr": "",
        "ja": ""
    }

    try:
        for chunk in transcription_chunks:
            translations["en"] += Translator(from_lang="es", to_lang="en").translate(chunk) + " "
            translations["it"] += Translator(from_lang="es", to_lang="it").translate(chunk) + " "
            translations["fr"] += Translator(from_lang="es", to_lang="fr").translate(chunk) + " "
            translations["ja"] += Translator(from_lang="es", to_lang="ja").translate(chunk) + " "
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traduciendo el texto: {str(e)}")

    print(f"Texto traducido a Inglés: {translations['en']}")
    print(f"Texto traducido a Italiano: {translations['it']}")
    print(f"Texto traducido a Francés: {translations['fr']}")
    print(f"Texto traducido a Japonés: {translations['ja']}")

    # 3. Generar audio traducido

    en_save_file_path = text_to_speach(translations["en"], "en")
    it_save_file_path = text_to_speach(translations["it"], "it")
    fr_save_file_path = text_to_speach(translations["fr"], "fr")
    ja_save_file_path = text_to_speach(translations["ja"], "ja")

    return en_save_file_path, it_save_file_path, fr_save_file_path, ja_save_file_path


def text_to_speach(text: str, language: str) -> str:

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=0.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = f"audios/{language}.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error creando el audio: {str(e)}")

    return save_file_path


web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone","upload"],
        type="filepath",
        label="Español"
    ),
    outputs=[
        gr.Audio(label="Inglés"),
        gr.Audio(label="Italiano"),
        gr.Audio(label="Francés"),
        gr.Audio(label="Japonés")
    ],
    title="Traductor de voz",
    description="Traductor de voz con IA a varios idiomas"
)

web.launch()
