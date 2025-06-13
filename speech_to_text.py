import os
import json
import tempfile # Required for handling credentials in production
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech

# --- Basic Setup ---
app = Flask(__name__)
CORS(app)
load_dotenv()

from flask_sock import Sock
sock = Sock(app)
def setup_google_credentials():
    google_creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if google_creds_json:
        # In a production environment, write the credentials to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_f:
            temp_f.write(google_creds_json)
            filepath = temp_f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = filepath
        print("✅ Google Credentials configured from GOOGLE_CREDENTIALS_JSON environment variable.")
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        # For local development where the path is already set
        print("✅ Google Credentials configured from GOOGLE_APPLICATION_CREDENTIALS path.")
    else:
        print("⚠️  WARNING: Google Cloud credentials not found. Speech-to-Text will fail.")

setup_google_credentials() # Call the setup function at startup

# --- API Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    print("✅ Gemini Model configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

try:
    # The speech client will now automatically use the credentials we set up above
    speech_client = speech.SpeechClient()
    print("✅ Google Cloud Speech-to-Text client configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Google Cloud Speech client: {e}")
    speech_client = None

# ... The rest of your code remains exactly the same ...

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
        cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_json_string)
        
        if "extracted_terms" not in result: result["extracted_terms"] = {}
        result["source_language"] = source_lang
        return result
    except Exception as e:
        print(f"❌ Gemini processing error: {e}")
        return {
            "error": "Failed to process text with Gemini.", "details": str(e),
            "final_english_text": transcript, "extracted_terms": {}, "source_language": source_lang
        }

@sock.route('/speech/<lang_code>')
def speech_socket(ws, lang_code):
    print(f"🟢 Client connected for language: {lang_code}. Starting Google STT stream...")
    if not speech_client:
        print("🔴 Speech client not available. Closing connection.")
        ws.close(reason=1011, message="Server-side speech client not configured.")
        return

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
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    def request_generator(websocket):
        try:
            while True:
                message = websocket.receive()
                if message is None: break
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get('type') == 'end_stream':
                        print("Generator: Received 'end_stream' signal.")
                        break
                else: 
                    yield speech.StreamingRecognizeRequest(audio_content=message)
        except Exception as e:
            print(f"Generator error: {e}")

    try:
        responses = speech_client.streaming_recognize(
            config=streaming_config,
            requests=request_generator(ws)
        )
        final_transcript = ""
        for response in responses:
            if not ws.connected: break
            if not response.results or not response.results[0].alternatives: continue
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            ws.send(json.dumps({ "type": "transcript", "is_final": result.is_final, "text": transcript }))
            if result.is_final: final_transcript += transcript + " "

        print(f"✅ Final Transcript for Gemini: {final_transcript}")
        if final_transcript.strip():
            gemini_result = get_gemini_extraction(final_transcript, lang_code)
            ws.send(json.dumps({ "type": "entities", "data": gemini_result }))
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        try: ws.send(json.dumps({ "type": "error", "message": f"Streaming Error: {e}" }))
        except: pass
    finally:
        print("🔴 Stream closed.")
        if ws.connected:
            try: ws.close()
            except: pass

# This block is for local testing only. Gunicorn (the production server) will not use it.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)