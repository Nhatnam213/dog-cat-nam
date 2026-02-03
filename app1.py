from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = tf.keras.models.load_model('model.h5')

# LẤY INPUT SHAPE TỪ MODEL
_, IMG_H, IMG_W, IMG_C = model.input_shape

def predict_image(img_path):
    img = Image.open(img_path)

    # Nếu model dùng grayscale
    if IMG_C == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = img.resize((IMG_W, IMG_H))
    img_array = np.array(img) / 255.0

    # Nếu grayscale thì thêm channel
    if IMG_C == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "🐶 Chó"
    else:
        return "🐱 Mèo"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            result = predict_image(image_path)

    return render_template('index1.html', result=result, image=image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
