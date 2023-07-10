import base64
import logging
import tracemalloc
from io import BytesIO

import cv2
import fire
import mediapipe
import numpy as np
import torch
import yaml
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from torchvision.transforms import Compose

from lib.models.pix2pix import Pix2Pix
from lib.transform import FaceCrop, ZeroPaddingResize

IMAGE_SIZE = 512

app = Flask(__name__)
cors = CORS(app, resources={r"/ganonymization/api/*": {"origins": "*"}})


@app.route('ganonymization/api/v1.0/landmarks/mediapipe', methods=['POST'])
async def face_to_landmarks():
    image_base64 = request.json['image']
    result = extract_landmarks_from_faces(image_base64)
    return jsonify({'landmarks': result})


@app.route('/ganonymization/api/v1.0/anonymize_landmarks', methods=['POST'])
async def landmarks_to_face():
    landmarks = request.json['landmarks']
    result = generate_face_from_landmarks(landmarks)
    return jsonify({'images': result})


@app.route('/ganonymization/api/v1.0/anonymize_faces', methods=['POST'])
async def face_to_face():
    image_base64 = request.json['image']
    result = extract_landmarks_from_faces(image_base64)
    result = generate_face_from_landmarks(result)
    return jsonify({'images': result})


def extract_landmarks_from_faces(image_base64):
    img = Image.open(BytesIO(base64.b64decode(image_base64)))
    faces = FaceCrop(True, True)(img)
    results = []
    for face in faces:
        face = ZeroPaddingResize(IMAGE_SIZE)(face)
        with mediapipe.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results.append(face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).multi_face_landmarks)
    return results


def generate_face_from_landmarks(landmarks):
    result = []
    for landmark in landmarks:
        # Generate the image
        with torch.no_grad():
            # XYZ-coordinates to 2D-image
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
            image_rows, image_cols, _ = image.shape
            x_min = min([coordinate['x'] for coordinate in landmark])
            x_max = max([coordinate['x'] for coordinate in landmark])
            y_min = min([coordinate['y'] for coordinate in landmark])
            y_max = max([coordinate['y'] for coordinate in landmark])
            aspect_ratio = (y_max - y_min) / (x_max - x_min)
            face_height = IMAGE_SIZE
            face_width = IMAGE_SIZE / aspect_ratio
            for coordinates in landmark:
                x_unnormalized = (coordinates['x'] - x_min) / (x_max - x_min)
                y_unnormalized = (coordinates['y'] - y_min) / (y_max - y_min)
                center = IMAGE_SIZE / 2
                x = x_unnormalized * face_width + center - face_width / 2
                y = y_unnormalized * face_height + center - face_height / 2
                cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)

            image = Image.fromarray(image)
            # 2D-image to synthesized image
            input_tensor = Compose(g.model.transforms_)(image)
            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = g.model.forward(input_tensor.to(g.model.device))
            output_tensor = torch.squeeze(output_tensor)
            output_np = output_tensor.add_(1).mul(128).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            output_image = Image.fromarray(output_np)
            buffered = BytesIO()
            output_image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result.append(base64_image)
    return result


def run_webapp(config_file: str):
    """
    Main entry point to start the web-application.
    :param config_file: The file containing all necessary configuration settings.
    """
    with open(config_file, 'r') as conf_file:
        config = yaml.safe_load(conf_file)
        # setup logging
        logging.basicConfig(level=config['logging']['level'])
        # create app
        if not (config is None):
            app.config.from_mapping(config)
        server_host = config['server']['host']
        server_port = config['server']['port']
        debug_mode = server_host in ['127.0.0.1', 'localhost']
        if debug_mode:
            tracemalloc.start()

        # Model
        model = Pix2Pix.load_from_checkpoint(config['model']['url'])
        model.eval()

        # Objects
        @app.before_request
        def setup_context():
            if not ('model' in g):
                g.model = model

        # start app
        app.run(host=server_host, port=server_port, debug=debug_mode)


if __name__ == '__main__':
    fire.Fire(run_webapp)
