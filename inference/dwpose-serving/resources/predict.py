import glob
import os
import sys
import time
from io import BytesIO
from PIL import Image
import base64
import json

from flask import Flask, Response, request

sys.path.append("/dwpose")

from dwpose import DWposeDetector

class DWPose:
    model = None
    
    # class method to load trained model and create an offline predictor
    @classmethod
    def create_model(cls):
        """load trained model"""
        try:
            model_dir = os.environ["SM_MODEL_DIR"]
        except KeyError:
            model_dir = "/opt/ml/model"

        cls.model = DWposeDetector(model_dir=model_dir)
        cls.model = cls.model.to(f"cuda")
        return cls.model
        
    @classmethod
    def get_model(cls):
        return cls.model

def encode_pil_image(pil_image, format="PNG"):
    # Create a BytesIO object
    buffered = BytesIO()
    
    # Save the image to the BytesIO object
    pil_image.save(buffered, format=format)
    
    # Get the byte value from the BytesIO object
    img_bytes = buffered.getvalue()
    
    # Encode the bytes to base64
    return base64.b64encode(img_bytes).decode('utf-8')

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def health_check():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully and crrate a predictor."""
    health = True
    
    if not DWPose.get_model():
        health = DWPose.create_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return Response(response="\n", status=status, mimetype="application/json")

dwpose_detector = DWPose.create_model()

@app.route("/invocations", methods=["POST"])
def inference():
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        return Response(response=result, status=415, mimetype="application/json")

    path = None
    try:
        content = request.get_json()
        images_path = []
        if "images" in content:
            encoded_images = content.pop("images")
            detect_resolution = content.pop("detect_resolution") if "detect_resolution" in content else 512
            image_resolution = content.pop("image_resolution") if "image_resolution" in content else 512
            
            kps_results = []
            for idx, encoded_image in enumerate(encoded_images):
                # Using a context manager to ensure the BytesIO stream is properly closed
                with BytesIO(base64.b64decode(encoded_image)) as f:
                    with Image.open(f) as input_image:
                        # Convert the image to 'RGB' only if it's not already in 'RGB' mode
                        if input_image.mode != 'RGB':
                            input_image = input_image.convert('RGB')
                        
                        print(input_image)
                        result, score = dwpose_detector(input_image, detect_resolution=detect_resolution, image_resolution=image_resolution,)
                        encoded_result = encode_pil_image(result)
                        kps_results.append(encoded_result)
            return Response(response=json.dumps(kps_results), status=200, mimetype="application/json")
        else:
            result = {"error": f"No images found"}
            return Response(response=result, status=500, mimetype="application/json")
    except Exception as e:
        print(str(e))
        result = {"error": f"Internal server error"}
        return Response(response=result, status=500, mimetype="application/json")
