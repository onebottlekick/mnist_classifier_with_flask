from flask import Flask, request, render_template
import numpy as np
import pickle
from PIL import Image

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

        prediction = model.predict(img)[0]
        
        return render_template('index.html', label=prediction)

if __name__ == '__main__':
    with open('./model/model.pkl', 'rb') as f:
            model = pickle.load(f)
    app.run(port=5000, debug=True)