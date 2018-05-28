
# coding: utf-8

# In[ ]:


from flask_api import FlaskAPI
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from sqlalchemy import text
import numpy as np
import os
import json
from PIL import Image
import base64
from io import BytesIO
from keras.models import Sequential
import keras.layers as l
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import pytesseract


# initialize sql-alchemy
db = SQLAlchemy()


app = FlaskAPI(__name__, instance_relative_config=True)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# In[ ]:


def neuralNet():
    model = Sequential(name="mlp")

    model.add(l.InputLayer([784]))
    model.add(l.Dense(256))
    model.add(l.Activation('relu'))

    model.add(l.Dense(128))
    model.add(l.Activation('relu'))
    model.add(l.Dense(26 , activation='softmax'))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

def nn_model2():
    model = Sequential()

    model.add(l.InputLayer([784]))
    model.add(l.Dense(500))
    model.add(l.Activation('relu'))

    model.add(l.Dense(500))
    model.add(l.Activation('relu'))
    
    model.add(l.Dense(128))
    model.add(l.Activation('relu'))
    model.add(l.Dense(62 , activation='softmax'))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

def cnn():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(l.Dense(128, activation='relu'))
    model.add(l.Dense(26, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# In[ ]:


alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet2=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S',
'T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


@app.route('/')
def show_all():
    return render_template('info.html')

@app.route('/get_prediction',methods=['POST'])
def predict():
    data = request.get_json()
    data__uri = data['url']
    w=np.loadtxt("weights.csv",delimiter=',')
    b=np.loadtxt("bias.csv",delimiter=',')
    dimensions = (28, 28)
    encoded_image = data__uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image))

    img = img.resize(dimensions, Image.ANTIALIAS)

    pixels = np.asarray(img, dtype='uint8')

    x = pixels[:,:,0]/255
    x = x.reshape(784,1)
    z=np.dot(x.T,w)+b
    pred_y= 1/(1+np.exp(-z))
    print(pred_y)
    print(pred_y.shape)
    print(np.amax(pred_y))
    q=pred_y
    pred=np.where(np.amax(pred_y,axis=1)==pred_y)
    return jsonify(int(pred[1])),200
@app.route('/get_prediction_from_NN',methods=['POST'])
def pred_neuralNet():
    model = neuralNet()
    model.load_weights("weights_keras.h5")
    data = request.get_json()
    data_uri = data['url']
    dimensions = (28, 28)
    encoded_image = data_uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image)).convert('L')
    img = img.resize(dimensions, Image.ANTIALIAS)
    pixels = np.asarray(img,dtype='float32')

    if pixels.ndim>2:
        x = (pixels[:,:,0]+pixels[:,:,1]+pixels[:,:,2])/3
        x = x/255
    else:
        x = pixels/255
    x = x.reshape(1,784)
    return jsonify(alphabet[int(model.predict_proba(x).argmax(axis=-1))].upper()),200
@app.route('/get_prediction_nn',methods=['POST'])
def pred_neuralnet():
    model = nn_model2()
    model.load_weights("weights_ocr.h5")
    data = request.get_json()
    data_uri = data['url']
    dimensions = (28, 28)
    encoded_image = data_uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image)).convert('L')
    img = img.resize(dimensions, Image.ANTIALIAS)
    pixels = np.asarray(img,dtype='float32')

    if pixels.ndim>2:
        x = (pixels[:,:,0]+pixels[:,:,1]+pixels[:,:,2])/3
        x = x/255
    else:
        x = pixels/255
    x = x.reshape(1,784)
    return jsonify(alphabet2[int(model.predict_proba(x).argmax(axis=-1))]),200
@app.route('/get_prediction_cnn',methods=['POST'])
def pred_cneuralnet():
    model = cnn()
    model.load_weights("weights.model")
    data = request.get_json()
    data_uri = data['url']
    dimensions = (28, 28)
    encoded_image = data_uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image)).convert('L')
    img = img.resize(dimensions, Image.ANTIALIAS)
    pixels = np.asarray(img,dtype='float32')

    if pixels.ndim>2:
        x = (pixels[:,:,0]+pixels[:,:,1]+pixels[:,:,2])/3
        x = x/255
    else:
        x = pixels/255
    x = x.reshape(1,28,28,1)
    return jsonify(alphabet[int(model.predict_proba(x).argmax(axis=-1))]),200
@app.route('/get_prediction2',methods=['POST'])
def pred_getp2():
    
    data = request.get_json()
    data_uri = data['url']
    dimensions = (28, 28)
    encoded_image = data_uri.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)

    img = Image.open(BytesIO(decoded_image))

    text = pytesseract.image_to_string(img, lang = 'eng')
    print(text)
    return jsonify(text),200

if __name__ == '__main__':
   app.debug = True
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port)
