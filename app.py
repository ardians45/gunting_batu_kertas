from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# --- PERUBAHAN DI SINI: Ubah nama file model yang dimuat ---
model = load_model('models/best_model_rps.keras')
labels = ['paper', 'rock', 'scissors']

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File tidak memiliki nama"}), 400
    
    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            processed_image = prepare_image(image, target_size=(150, 150))
            prediction_probabilities = model.predict(processed_image)[0]
            predicted_index = np.argmax(prediction_probabilities)
            predicted_class = labels[predicted_index]
            return jsonify({
                "prediksi": predicted_class,
                "probabilitas": prediction_probabilities.tolist()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File tidak valid"}), 400

# Baris untuk menjalankan server saat development
if __name__ == "__main__":
    app.run(debug=True)