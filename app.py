from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF info logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)
CORS(app)

# ── Model definition (must match Colab exactly) ───────────────────────────────

class GhostConv(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(GhostConv, self).__init__(**kwargs)
        self.primary_conv    = layers.Conv2D(filters // 2, (3, 3), padding='same', activation='relu')
        self.cheap_operation = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')

    def call(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return tf.concat([x1, x2], axis=-1)


def attention_block(x):
    channels = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Dense(channels // 8, activation='relu')(avg_pool)
    avg_pool = layers.Dense(channels, activation='sigmoid')(avg_pool)
    scale    = layers.Reshape((1, 1, channels))(avg_pool)
    return layers.Multiply()([x, scale])


def build_model():
    inputs = keras.Input(shape=(224, 224, 1))

    x = GhostConv(32)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = GhostConv(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = attention_block(x)

    x = GhostConv(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = GhostConv(256)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name='GhostAttentionCNN')


# ── Load model ─────────────────────────────────────────────────────────────────
# Supports .h5, .keras, or SavedModel directory — just rename MODEL_PATH below

MODEL_PATH = "ghost_attention_bone_cancer.h5"

model = None

# Try full model load first (saves architecture + weights together)
for path in [MODEL_PATH,
             MODEL_PATH.replace('.h5', '.keras'),
             MODEL_PATH.replace('.h5', '')]:
    if os.path.exists(path):
        try:
            model = keras.models.load_model(
                path, custom_objects={"GhostConv": GhostConv})
            print(f"✔ Full model loaded from: {path}")
            break
        except Exception as e:
            print(f"Full load failed for {path}: {e}")

# Fallback: build architecture and load weights only
if model is None:
    model = build_model()
    weights_paths = [MODEL_PATH,
                     MODEL_PATH.replace('.h5', '.weights.h5')]
    for wp in weights_paths:
        if os.path.exists(wp):
            model.load_weights(wp)
            print(f"✔ Weights loaded from: {wp}")
            break
    else:
        print("WARNING: No model file found. Predictions will be random.")

# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ..., np.newaxis]                # (1,224,224,1)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        arr  = preprocess(file.read())
        prob = float(model.predict(arr, verbose=0)[0][0])  # sigmoid → [0,1]

        # prob = P(Cancer) if you labelled Cancer=1 in Colab
        # If Normal=1 in your dataset, swap the two lines below
        cancer_prob = 1.0 - prob
        normal_prob = prob

        if cancer_prob >= 0.5:
            prediction = "Cancer"
            confidence = round(cancer_prob * 100, 1)
        else:
            prediction = "Normal"
            confidence = round(normal_prob * 100, 1)

        return jsonify({
            'success':    True,
            'prediction': prediction,
            'confidence': confidence,
            'all_probs': {
                'Cancer': round(cancer_prob * 100, 2),
                'Normal': round(normal_prob * 100, 2)
            }
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status':    'healthy',
        'framework': 'TensorFlow ' + tf.__version__,
        'model':     MODEL_PATH
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)