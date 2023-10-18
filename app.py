import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from utils import predict_dog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'



@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        predicted_class = predict_dog(filename)
        print("preidcted_class:", predicted_class)
        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)