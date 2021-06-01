from flask import Flask, request
import os, json

import numpy as np

app = Flask(__name__)

@app.route('/', methods=["GET"])
def index():
    data = {"status": 200, "data": "Füdery Nutrition Assistant"}
    umur = {
        "Anjar": 21,
        "Tomi": 21,
        "Roni": 22,
        "Joni": 24,
    }
    data["val_umur"] = umur
    return json.dumps(data)

@app.route('/input/<int:id>', methods=['GET'])
def input_id(id):
    parameter = str(id)
    return parameter

@app.route('/input/post/', methods=['POST'])
def input_post():
    args1 = request.args.get("nama")
    return args1

if __name__ == '__main__':
    # app.run()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))