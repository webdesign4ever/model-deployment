import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from keras.models import load_model
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib

app = Flask(__name__)

# Load the saved model
MODEL_PATH = 'model'
loaded_model = load_model(MODEL_PATH)

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def prepare_img1(image_path):
    img_size = (128, 128)  # Adjust to match your model's input size
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format if needed
    img = cv2.resize(img, img_size)  # Resize the image
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, *img_size, 3)      # Reshape to (1, 128, 128, 3)
    return np.array(img)

def make_gradcam_heatmap(img_arr,model,last_conv_layer_name,pred_index=None):
    grad_model=tf.keras.models.Model(
        [model.input],[model.get_layer(last_conv_layer_name).output,model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output,preds=grad_model(img_arr)

        if pred_index is None:
            pred_index=tf.argmax(preds[0])

        class_channel=preds[:,pred_index]

    grads=tape.gradient(class_channel,last_conv_layer_output)
    pooled_grads=tf.reduce_mean(grads,axis=(0,1,2))
    last_conv_layer_output=last_conv_layer_output[0]

    heatmap=last_conv_layer_output @ pooled_grads[...,tf.newaxis]
    heatmap=tf.squeeze(heatmap)
    heatmap=tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_and_display_gradcam(img_path,heatmap,cam_path="cam.jpg",alpha=0.4):
    img=img_to_array(load_img(img_path))
    heatmap=np.uint8(255*heatmap)
    jet = matplotlib.colormaps.get_cmap("jet")
    jet_colors=jet(np.arange(256))[:,:3]
    jet_heatmap=jet_colors[heatmap]

    jet_heatmap=keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap=jet_heatmap.resize((img.shape[1],img.shape[0]))
    jet_heatmap=keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img=jet_heatmap * alpha + img
    superimposed_img=keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

@app.route('/')
def index():
    return render_template('index.html')

# Global variable to store the image path
image_path_global = None

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    global image_path_global

    if request.method == 'POST':
        if image_path_global is None:
            return jsonify({'error': 'No image available. Run the predict API first.'})
        
        # Use the previously saved image path
        image_path = image_path_global
        
        # Prepare the image
        img_array = prepare_img1(image_path)

        # Generate GradCAM heatmap
        last_conv_layer_name = "conv2d_1"  # Replace with your layer name
        heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
           
        # Save and display the GradCAM heatmap
        cam_path = os.path.join('heatmap', 'heatmap.jpg')
        save_and_display_gradcam(image_path, heatmap, cam_path)
        return jsonify({'heatmap_path': cam_path})
    else:
        return jsonify({'error': 'Invalid request'})
    
@app.route('/heatmap/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory('heatmap', filename) 

@app.route('/predict', methods=['POST'])
def predict():
    global image_path_global
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            image_path_global = os.path.join('uploads', 'image.jpg')
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file.save(image_path_global)
            class_names = ['fake', 'real']
            processed_image = prepare_image(image_path_global)
            processed_image=processed_image.reshape(-1,128,128,3)
            prediction = loaded_model.predict(processed_image)
            y_pred_class = np.argmax(prediction, axis = 1)[0]
            print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(prediction) * 100:0.2f}')
            class_name = class_names[y_pred_class]
            confidence_score = float(np.amax(prediction) * 100)
            prediction_result = {
                'prediction': class_name,
                'confidence': confidence_score  
            }
            return jsonify(prediction_result)
    else:
        return jsonify({'error': 'Invalid request'})
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(debug=True)
