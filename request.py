#import requests

#url = 'http://localhost:5000/predict_api'
#r = requests.post(url,json={'street.jpg'})

#data = open('street.jpg','rb').read()
#r = requests.post(url,data=data)

#requests.post(url, files=files)

#print(r.json())

import base64
import json                    
from PIL import Image
import requests
import cv2
import io
import numpy as np

api = 'http://localhost:5000/predict_api'
image_file = 'street.jpg'

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
  
payload = json.dumps({"image": im_b64})
response = requests.post(api, data=payload, headers=headers)
try:
    data = response.json()
    final_output = data['output']
    #final_b64 = data.json['output']
    #final_bytes = base64.b64decode(final_b64.encode('utf-8'))
    #final_img = Image.open(io.BytesIO(final_bytes))
    img_bytes = base64.b64decode(final_output.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)
    cv2.imwrite('final-output.jpg', img_arr)
    #print(data)                
except requests.exceptions.RequestException:
    print(response.text)