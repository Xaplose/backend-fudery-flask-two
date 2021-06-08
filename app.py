from flask import Flask, request
from flask_cors import CORS
from model import predict
import json

UPLOAD_FOLDER = '/upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app, resources=r'/*')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/input/<int:id>', methods=['GET'])
def input_id(id):
    model = tf.keras.models.load_model('image-classification-7.h5')
    return model.summary()

@app.route('/predict', methods=['GET','POST'])
def predict_images():

    data = request.files.get("file")
    if data == None:
        return 'Got Nothing'
    else:
        prediction = predict(data)

    return json.dumps(str(prediction))