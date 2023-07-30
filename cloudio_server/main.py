import datetime
import time

from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
import os
import io
import cv2
from urllib.parse import unquote
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import requests
import tifffile
from goes_18_predictor import predict as geo_predict
from edgde_detection import predict_edge
from landsat_downloader import EarthEngineExporter
from geos_images_downloader import GeosEngineExporter
from landsat_preview import BandImageExporter
from landset_cloud_predictor import predict as landset_predict
from flask import Flask, request, jsonify
import base64
from bson import ObjectId

app = Flask(__name__)
load_dotenv('.env')
CORS(app,supports_credentials=True)

# MongoDB connection
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['cloudio']
collection_images = db['images']
collection_models = db['models']
users_collection = db["users"]

@app.route("/login", methods=["GET"])
def login():
    # Get the user credentials from the request
    userId = request.args.get("userId")
    password = request.args.get("password")

    # Query the MongoDB collection for the user
    user = users_collection.find_one({"userId": userId})

    if user and user["password"] == password:
        # Authentication successful
        response = {
            "user": {
                "userId": user["userId"],
                "userName": user["userName"],
                "userType": user["userType"]
            },
            "accessToken": "your_access_token"
        }
        return jsonify(response), 200
    else:
        # Authentication failed
        return jsonify({"message": "Invalid credentials"}), 401

@app.route('/landset_download', methods=['GET','POST'])
def download_and_preview_landset():
    cords = request.json
    exporter = EarthEngineExporter()
    exporter.set_coordinates(cords)
    export_directory, file_name_preffix = exporter.export_images()
    exporter = BandImageExporter()
    export_directory = f"{os.getenv('STORAGE_PATH')}/{export_directory}/"
    exporter.set_export_directory(export_directory)
    retries = 15
    plot_path=""
    while (retries>0):
        try:
            plot_path = exporter.plot_band_image(file_name_preffix.getInfo(), "B3")
            retries = 0
            break
        except Exception as e:
            print("waiting for file to be uploaded")
            retries = retries -1
            time.sleep(10)
    time.sleep(30)
    plot_path = exporter.plot_band_image(file_name_preffix.getInfo(), "B3")
    # Read the image file
    # Read the image file
    with open(plot_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Create the response JSON
    response = {
        'image': image_base64,
        'export_directory': export_directory,
        'image_prefix': file_name_preffix.getInfo()
    }

    # Return the response as JSON
    return jsonify(response)

@app.route('/geos_download', methods=['GET','POST'])
def download_and_preview_geos():
    cords = request.json
    exporter = GeosEngineExporter()
    exporter.set_coordinates(cords)
    export_directory, file_name_preffix = exporter.export_images()
    exporter = BandImageExporter()
    export_directory = f"{os.getenv('STORAGE_PATH')}/{export_directory}/"
    exporter.set_export_directory(export_directory)
    retries = 15
    plot_path=""
    while (retries>0):
        try:
            plot_path = exporter.plot_geos_image(file_name_preffix)
            retries = 0
            break
        except Exception as e:
            print("waiting for file to be uploaded")
            retries = retries -1
            time.sleep(10)
    time.sleep(30)
    plot_path = exporter.plot_geos_image(file_name_preffix)
    # Read the image file
    with open(plot_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Create the response JSON
    response = {
        'image': image_base64,
        'export_directory': export_directory,
        'image_prefix': file_name_preffix
    }

    # Return the response as JSON
    return jsonify(response)


@app.route('/landset_predict_get', methods=['POST'])
def landset_predict_get():
    # Get algorithm name and image location from the request
    request_data = request.json
    image_edge_path = ""
    edge_masked_base64 = "empty"
    if (request_data["img_source"] == 'landset_download'):
        image_path, masked_path = landset_predict(request_data["export_directory"], request_data["image_prefix"])
    else:
        image_path, masked_path = geo_predict(request_data["export_directory"], request_data["image_prefix"])

    if request_data["algo_name"] == "edge" and request_data["img_source"] == 'landset_download':
        image_edge_path = predict_edge(request_data["export_directory"], request_data["image_prefix"])
        # Read the masked image and encode it as base64
        with open(image_edge_path, 'rb') as f:
            masked_base64 = base64.b64encode(f.read()).decode('utf-8')
    elif request_data["algo_name"] == "edge":
        image_edge_path = predict_edge(request_data["export_directory"], request_data["image_prefix"],"geos")
        # Read the masked image and encode it as base64
        with open(image_edge_path, 'rb') as f:
            masked_base64 = base64.b64encode(f.read()).decode('utf-8')
    else:
        # Read the masked image and encode it as base64
        with open(masked_path, 'rb') as f:
            masked_base64 = base64.b64encode(f.read()).decode('utf-8')

    print(image_path)
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')





    # Create a JSON response with the base64-encoded images
    response = {
        'resized_image': image_base64,
        'masked_image': masked_base64,
    }

    return jsonify(response)

# Endpoint for predicting and saving an image
@app.route('/predict', methods=['POST'])
def predict():
    # Get algorithm name and image location from the request
    algorithm_name = request.form.get('algorithm_name')
    image_location = request.form.get('image_file')
    if (algorithm_name == 'geoset'):
        image_path = geo_predict()
    else:
        image_path = landset_predict()
    print(image_path)
    resized_image = convert_predicted_tiff(image_path)
    # Create an in-memory stream to hold the JPEG data
    image_stream = io.BytesIO()
    # Save the resized image as JPEG to the stream
    retval, encoded_image = cv2.imencode('.jpg', resized_image)
    if not retval:
        return {'error': 'Failed to encode image as JPEG'}

    # Write the encoded image to the stream
    image_stream.write(encoded_image)

    # Seek to the beginning of the stream
    image_stream.seek(0)

    # Return the JPEG image as a response
    return send_file(image_stream, mimetype='image/jpeg')

@app.route('/load_models', methods=['GET'])
def load_models():
    models = collection_models.find()
    model_list = []
    for model in models:
        model_list.append({
            '_id': str(model['_id']),
            'modelName': model['modelName'],
            'modelType': model['modelType']
        })
    return {'models': model_list}

@app.route('/upload_models', methods=['POST'])
def upload_models():
    model_name = request.form.get('modelName')
    model_type = request.form.get('modelType')
    destination_path = f"{os.getenv('STORAGE_PATH')}/uploaded_models"

    if 'modelFile' not in request.files:
        return {'message': 'No model file provided.'}, 400

    model_file = request.files['modelFile']
    if model_file.filename == '':
        return {'message': 'No model file selected.'}, 400

    # Save the model file to the destination path
    model_file.save(os.path.join(destination_path, model_file.filename))

    model_data = {
        'modelName': model_name,
        'modelType': model_type,
        'destinationPath': destination_path
    }
    result = collection_models.insert_one(model_data)

    return {'message': 'Model uploaded successfully.', 'model_id': str(result.inserted_id)}


@app.route('/load_images', methods=['GET'])
def load_images():
    images = collection_images.find()
    images_list = []
    for image in images:
        images_list.append({
            'id': str(image['_id']),
            'image_name': image['image_name'],
            'algo_name': image['algo_name'],
            'export_directory': image['export_directory'],
            'image_prefix': image['image_prefix'],
            'predicted_image': image['predicted_image'],
            'predicted_masked_image': image['predicted_masked_image'],
            'update_date': image['update_date'],
            'coordinates': image['coordinates'],
            'downloaded_image': image['downloaded_image']
        })
    return {'images': images_list}


@app.route('/save_results', methods=['POST'])
def save_results():
    image_name = request.json['image_name']
    algo_name = request.json['algo_name']
    export_directory = request.json['export_directory']
    image_prefix = request.json['image_prefix']
    predicted_image = request.json['predicted_image']
    predicted_masked_image = request.json['predicted_masked_image']
    coordinates = request.json['coordinates']
    downloaded_image = request.json['downloaded_image']
    date = str(datetime.datetime.now())
    image_data = {
    'image_name': image_name,
    'algo_name': algo_name,
    'export_directory': export_directory,
    'image_prefix': image_prefix,
    'predicted_image': predicted_image,
    'predicted_masked_image': predicted_masked_image,
    'downloaded_image':downloaded_image,
    'update_date': date,
        'coordinates': coordinates
    }
    result = collection_images.insert_one(image_data)

    return {'message': 'Model uploaded successfully.', 'model_id': str(result.inserted_id)}


@app.route('/delete_image/<string:image_id>', methods=['DELETE'])
def delete_image(image_id):
    result = collection_images.delete_one({'_id': ObjectId(image_id)})

    if result.deleted_count > 0:
        return jsonify({'success': True, 'message': 'Image deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Image not found'})


if __name__ == '__main__':
    app.run(debug=True)
