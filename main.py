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
    model = tf.keras.models.load_model('./model/model.h5')
    app.run(port=5000, debug=True)