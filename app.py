from flask import Flask, render_template, request, send_file, send_from_directory, jsonify, abort
from werkzeug import secure_filename
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image
from yolov3.configs import *
import io
import json                    
import base64                  
import logging             
from PIL import Image



if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE)
load_yolo_weights(yolo, Darknet_weights) # use Darknet weights


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def recieve_file():
   if request.method == 'POST':
      f = request.files['file']
      img = secure_filename((f.filename))
      f.save(secure_filename(f.filename))
      image,cl = detect_image(yolo, img, '', input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
      print(cl)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      cv2.imwrite('street-output.jpg', image)
      #return render_template('output.html',yolo_output  = 'street-output.jpg' )
      #return send_from_directory(directory='SIMPLEIMAGEDEPLOYMENTTEST', filename='street-output.jpg')
      return send_file('Street-output.jpg', as_attachment=True)

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    #data = request.get_json(force= True)
    #image = request.get_data()
    #print(image)
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)


    # process your img_arr here    
    
    cv2.imwrite('test.jpg', img_arr)
    output_path = 'test.jpg'
    image,cl = detect_image(yolo, output_path, '', input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('test-output.jpg', image)
    # access other keys of json
    # print(request.json['other_key'])
    output_file = 'test-output.jpg'
    with open(output_file, "rb") as f:
        output_bytes = f.read()        
    output_b64 = base64.b64encode(output_bytes).decode("utf8")
    
    result_dict = {'output': output_b64}
    return result_dict
    
		
if __name__ == '__main__':
   app.run(debug = True)