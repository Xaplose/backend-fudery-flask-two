import tensorflow as tf
import base64
import io
import numpy as np
from PIL import Image
from keras.preprocessing import image

model = tf.keras.models.load_model('image-classification-7.h5')
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    return img

def predict(data):
    img = Image.open(data)
    processed_image = preprocess_image(img, target_size=(224, 224))

    images = np.vstack([processed_image])
    classes = model.predict(images, batch_size=10)
    label = np.where(classes[0] > 0.5, 1,0)
    data = [
        'bibimbap', "chicken_wings", "churros", "cup_cakes",
        "donuts","dumplings","edamame","fish_and_chips","french_fries",
        "fried_rice","frozen_yogurt", "hamburger","ice_cream","lasagna",
        "macaroni_and_cheese","macarons","omelette","onion_rings","pancakes",
        "pizza","ramen","spahetti_bolognese","spahetti_carbonara",
        "steak","takoyaki","waffles"
    ]
    return str(label) + str(data)