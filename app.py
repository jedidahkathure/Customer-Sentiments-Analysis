import pickle
import re
from flask import Flask, request, jsonify

# Initialize app
app = Flask(__name__)

# --- Global Variables for Assets ---
model = None
vectorizer = None
label_encoder = None

# --- Text Preprocessing Function (MUST match training script) ---
def clean_text(text: str) -> str:
    """Matches the simple text cleaning used during model training."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # Remove all non-alphabetic characters except spaces (matching the training clean_text)
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- 1. Load Model and Preprocessing Tools ---
try:
    # 1. Load the DICTIONARY containing the model and the label encoder
    # Using user's file name: lr_modell.pkl
    with open("lr_modell.pkl", "rb") as f:
        model_assets = pickle.load(f)
        
    # **CRITICAL FIX: Extract the actual model object and label encoder from the dictionary**
    # The dictionary keys are 'model' and 'encoder' as defined in the training script.
    model = model_assets.get('model')
    label_encoder = model_assets.get('encoder')

    # Load TF-IDF vectorizer (Using user's file name: tfidf_vectorizerl.pkl)
    with open("tfidf_vectorizerl.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    if model is None or label_encoder is None or vectorizer is None:
        raise Exception("One or more required assets ('model', 'encoder', or 'vectorizer') were not loaded or found inside the .pkl files.")

    print("✅ Model, Vectorizer, and Label Encoder loaded successfully.")

except FileNotFoundError:
    print("FATAL ERROR: One or more deployment files (.pkl) not found. Check file names: lr_modell.pkl, tfidf_vectorizerl.pkl")
    # Set assets to None to prevent server startup if files are missing
    model = None
    vectorizer = None
    label_encoder = None
except Exception as e:
    print(f"FATAL ERROR during asset loading: {e}")
    model = None
    vectorizer = None
    label_encoder = None

@app.route("/")
def home():
    if model and vectorizer and label_encoder:
        return "✅ Sentiment Analysis API is running and assets are loaded!"
    else:
        return "❌ Sentiment Analysis API failed to load assets. Check console for details.", 503


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None or label_encoder is None:
        return jsonify({"error": "Model assets not loaded. Check server startup logs."}), 503

    try:
        data = request.get_json()
        raw_text = data.get("text", "")

        if not raw_text:
            return jsonify({"error": "No text provided"}), 400
        
        # 1. Preprocess text (CRITICAL STEP)
        cleaned_text = clean_text(raw_text)

        # 2. Transform text to vector
        X = vectorizer.transform([cleaned_text])
        
        # 3. Predict numerical label
        prediction_encoded = model.predict(X)[0]
        
        # 4. Decode numerical prediction using the LabelEncoder (CRITICAL STEP)
        # prediction_encoded is an integer (e.g., 0, 1, 2). We must convert it back to a label (e.g., 'negative').
        sentiment_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Return the human-readable sentiment label
        response_data = {
            "text": raw_text,
            "sentiment": str(sentiment_label)
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Prediction error occurred: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
