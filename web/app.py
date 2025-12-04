# frontend/app.py
from flask import Flask, render_template, request
import torch
import librosa
import os

# Make sure your models folder is accessible
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "models"))

from cnn_model import CNNEmotionClassifier, extract_features_from_waveform

# Flask app
app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN model
model = CNNEmotionClassifier()
model_path = os.path.join(ROOT, "models", "best_cnn_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Labels
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                file.save(tmp.name)
                y, sr = librosa.load(tmp.name, sr=None)
                x = extract_features_from_waveform(y,sr)

            with torch.no_grad():
                logits = model(x)
                pred_idx = torch.argmax(logits, dim=1).item()
                result = LABELS[pred_idx]

    return render_template("index.html", result=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
