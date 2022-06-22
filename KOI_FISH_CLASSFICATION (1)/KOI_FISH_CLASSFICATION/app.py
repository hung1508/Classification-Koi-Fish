
from flask import Flask, flash, request, redirect, url_for, render_template, Response

from camera import Video
import urllib.request
import os
from werkzeug.utils import secure_filename
from utils import test
from model import MyModel
from dataloader import transformer
from hyperparameters import classes, model_path, device
from lib import torch

app = Flask(__name__)
model = MyModel().to(device)
model.load_state_dict(torch.load(os.path.join(model_path,"best_model.pth"), map_location=torch.device(device)))
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')

def gen_camera(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')
@app.route('/video')
def video():
    return Response(gen_camera(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/', methods=['POST'])
def upload_image():
    global img2Pred
    global filename
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img2Pred = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


# Gọi hàm xử lý chỗ này
@app.route('/predict', methods=['POST'])
def predict_image():
    global prediction
    img = img2Pred 
    prediction = test(model,img,classes,transformer)
    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run()