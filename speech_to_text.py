import os
import json
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech
from google.oauth2 import service_account
from flask_sock import Sock

# --- App Setup ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
load_dotenv()  # Load .env for local testing

@app.route('/')
def health_check():
    return "OK", 200

# --- Gemini Setup ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    print("✅ Gemini Model configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

# --- Google Cloud Speech Client Setup ---
speech_client = None
try:
    google_creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")

    if google_creds_json_str:
        creds_info = json.loads(google_creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        speech_client = speech.SpeechClient(credentials=credentials)
        print("✅ Google Speech Client configured from GOOGLE_CREDENTIALS_JSON.")
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        speech_client = speech.SpeechClient(credentials=credentials)
        print("✅ Google Speech Client configured from GOOGLE_APPLICATION_CREDENTIALS path.")

    else:
        print("⚠️  WARNING: No Google Cloud credentials found. STT will fail.")
except Exception as e:
    print(f"❌ FATAL: Error configuring Google Cloud Speech client: {e}")
    speech_client = None

# --- Gemini Text Extraction ---
def get_gemini_extraction(transcript, source_lang):
    if not gemini_model:
        return {"error": "Gemini model not configured."}
    if not transcript.strip():
        return {"error": "Cannot process empty transcript."}

    source_language_full_name = "English" if source_lang == "en" else "Malayalam"

    smart_prompt = f"""
    You are an advanced medical transcription AI. Your input is raw text from a speech-to-text system.
    The source language of the text is {source_language_full_name}.

    Your tasks are:
    1. If the input is in Malayalam, translate it to high-quality medical English.
    2. If the input is in English, clean it up and normalize medical terminology.
    3. Extract key medical information into structured JSON format.

    Raw Transcript Input:
    "{transcript}"

    Required JSON Output Format:
    {{
      "final_english_text": "The fully translated and normalized English text",
      "extracted_terms": {{
        "Medicine Names": [], "Dosage & Frequency": [], "Diseases / Conditions": [],
        "Symptoms": [], "Medical Procedures / Tests": [], "Duration": [], "Doctor's Instructions": []
      }},
      "source_language": "{source_lang}"
    }}

    Important Rules:
    - Always output valid JSON. The "source_language" in the JSON must be "{source_lang}".
    - Extract ALL medical terms you can find. If a category is empty, use an empty array [].
    """
    try:
        response = gemini_model.generate_content(smart_prompt)
        cleaned_json = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_json)
        result["source_language"] = source_lang
        if "extracted_terms" not in result:
            result["extracted_terms"] = {}
        return result
    except Exception as e:
        print(f"❌ Gemini processing error: {e}")
        return {
            "error": "Failed to process text with Gemini.",
            "details": str(e),
            "final_english_text": transcript,
            "extracted_terms": {},
            "source_language": source_lang
        }

# --- WebSocket for Live Transcription ---
@sock.route('/speech/<lang_code>')
def speech_socket(ws, lang_code):
    if not speech_client:
        print("🔴 Speech client not available. Closing connection.")
        ws.close(reason=1011, message="Speech client not configured.")
        return

    print(f"🟢 Client connected with language: {lang_code}")

    model_config = {
        'ml': {"language_code": "ml-IN", "model": "latest_long"},
        'en': {"language_code": "en-US", "model": "medical_dictation"}
    }
    selected_config = model_config.get(lang_code, model_config['en'])

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        **selected_config
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    def request_generator(websocket):
        try:
            while True:
                message = websocket.receive()
                if message is None:
                    break
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get('type') == 'end_stream':
                        print("🔁 End of stream signal received.")
                        break
                else:
                    yield speech.StreamingRecognizeRequest(audio_content=message)
        except Exception as e:
            print(f"⚠️ Generator error: {e}")

    try:
        responses = speech_client.streaming_recognize(
            config=streaming_config,
            requests=request_generator(ws)
        )

        final_transcript = ""
        for response in responses:
            if not ws.connected:
                break
            if not response.results or not response.results[0].alternatives:
                continue
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            ws.send(json.dumps({ "type": "transcript", "is_final": result.is_final, "text": transcript }))
            if result.is_final:
                final_transcript += transcript + " "

        print(f"✅ Final Transcript: {final_transcript}")
        if final_transcript.strip():
            gemini_result = get_gemini_extraction(final_transcript, lang_code)
            ws.send(json.dumps({ "type": "entities", "data": gemini_result }))
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        try:
            ws.send(json.dumps({ "type": "error", "message": f"Streaming Error: {e}" }))
        except:
            pass
    finally:
        print("🔴 Stream closed.")
        if ws.connected:
            try:
                ws.close()
            except:
                pass

# --- Local Dev Entry Point ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
