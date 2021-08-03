from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_predict():
    if request.method == "POST":

        file = request.files['image']
        if not file:
            return render_template('index.html', label='no files')
        
        img = np.array(Image.open(file).resize((28, 28)).convert('L')).reshape(1, -1)

        prediction = model.predict(img).argmax()
        
        return render_template('index.html', label=prediction)

if __name__ == '__main__':
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.load_weights('checkpoint/cp.ckpt')
    app.run(port=5000, debug=True)