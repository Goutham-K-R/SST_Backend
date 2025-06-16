# Apply gevent monkey-patching at the very top
import gevent.monkey
gevent.monkey.patch_all()

import os
import json
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech
from google.oauth2 import service_account
from flask_sock import Sock

# --- App Setup ---
app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Set default log level
CORS(app)
sock = Sock(app)
load_dotenv()

@app.route('/')
def health_check():
    return "OK", 200

# --- Service Configurations ---
gemini_model = None
speech_client = None

try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    app.logger.info("Gemini Model configured successfully.")
except Exception as e:
    app.logger.error(f"Error configuring Gemini: {e}")

try:
    creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if creds_json_str:
        creds_dict = json.loads(creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        speech_client = speech.SpeechClient(credentials=credentials)
        app.logger.info("Google Speech Client configured from GOOGLE_CREDENTIALS_JSON.")
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        speech_client = speech.SpeechClient(credentials=credentials)
        app.logger.info("Google Speech Client configured from GOOGLE_APPLICATION_CREDENTIALS path.")
    else:
        app.logger.warning("No Google credentials found. STT will not work.")
except Exception as e:
    app.logger.error(f"FATAL: Error configuring Google Cloud Speech client: {e}", exc_info=True)


# --- Gemini Extraction Logic ---
def get_gemini_extraction(transcript, source_lang):
    if not gemini_model:
        return {"error": "Gemini model not configured."}
    if not transcript.strip():
        return {"error": "Empty transcript."}

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

    Rules:
    - Output valid JSON only.
    - Use empty arrays if a category has no data.
    - Always include 'source_language': "{source_lang}"
    """
    try:
        response = gemini_model.generate_content(smart_prompt)
        cleaned_text = response.text.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(cleaned_text)
        result.setdefault("extracted_terms", {})
        result["source_language"] = source_lang
        return result
    except Exception as e:
        app.logger.error(f"Gemini processing error: {e}", exc_info=True)
        return {
            "error": "Gemini failed to process input.",
            "details": str(e),
            "final_english_text": transcript,
            "extracted_terms": {},
            "source_language": source_lang
        }

# --- WebSocket Speech Handler ---
@sock.route('/speech/<lang_code>')
def speech_socket(ws, lang_code):
    try:
        if not speech_client:
            app.logger.warning("WebSocket connection failed: Speech client not configured.")
            ws.close(reason=1011, message="Server-side speech client not configured.")
            return

        app.logger.info(f"WebSocket connected for lang_code: {lang_code}")

        model_config = {'ml': {"language_code": "ml-IN", "model": "latest_long"}}
        default_config = {"language_code": "en-US", "model": "medical_dictation"}
        selected_config = model_config.get(lang_code, default_config)
        
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
                    if message is None: break
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get("type") == "end_stream":
                            app.logger.info("Stream ended by client signal.")
                            break
                    else:
                        yield speech.StreamingRecognizeRequest(audio_content=message)
            except Exception as e:
                app.logger.error(f"Error in request_generator: {e}")

        responses = speech_client.streaming_recognize(config=streaming_config, requests=request_generator(ws))

        final_transcript = ""
        for response in responses:
            if not ws.connected: break
            if not response.results or not response.results[0].alternatives: continue
            
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            ws.send(json.dumps({"type": "transcript", "is_final": result.is_final, "text": transcript}))
            
            if result.is_final:
                final_transcript += transcript + " "

        app.logger.info(f"Final Transcript received: '{final_transcript.strip()}'")
        if final_transcript.strip():
            gemini_result = get_gemini_extraction(final_transcript, lang_code)
            ws.send(json.dumps({"type": "entities", "data": gemini_result}))

    except Exception as e:
        app.logger.error(f"Unhandled exception in WebSocket handler: {e}", exc_info=True)
        try:
            if ws.connected:
                ws.send(json.dumps({"type": "error", "message": f"An unexpected server error occurred: {e}"}))
        except:
            pass # Client might already be disconnected
    finally:
        app.logger.info("WebSocket connection closing.")
        if ws.connected:
            try: ws.close()
            except: pass

if __name__ == '__main__':
    # This block is for local development only, not used by Gunicorn
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)