import torch
import requests
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# ✅ Load Model & Tokenizer
MODEL_PATH = "conversational_medical_model"
MODEL_RELEASE_URL = "https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO/releases/download/v3.0/model(1).safetensors"

def download_model():
    print("Downloading model...")
    response = requests.get(MODEL_RELEASE_URL)
    with open("model.safetensors", "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
except:
    download_model()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Prediction Function
def predict_disease(symptoms):
    input_text = f"Patient reports symptoms: {symptoms}. What is the likely diagnosis and recommended treatment?"
    inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Speech-to-Text (Audio Input)
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# ✅ Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "audio" in data:
            text_input = speech_to_text(data["audio"])
        else:
            text_input = data["text"]

        prediction = predict_disease(text_input)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
