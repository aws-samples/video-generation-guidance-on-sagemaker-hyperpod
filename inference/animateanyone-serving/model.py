import torch
torch.manual_seed(1234)

from dataclasses import dataclass
from typing import List, Optional
from djl_python import Input, Output

from PIL import Image
import base64
from io import BytesIO
import json
import os

import numpy as np
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection
from torchvision import transforms

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

import boto3
from boto3.s3.transfer import TransferConfig
from urllib.parse import urlparse
from pathlib import Path

def download_from_s3_uri(s3_uri, local_file_path):
    # Parse the S3 URI
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    s3_file_path = parsed_uri.path.lstrip('/')

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Download the file
    s3.download_file(bucket_name, s3_file_path, local_file_path)
    
def upload_video_to_s3(local_file_path, s3_uri):
    # Parse the S3 URI
    parsed_uri = urlparse(s3_uri)
    bucket_name = parsed_uri.netloc
    s3_file_path = parsed_uri.path.lstrip('/')

    s3 = boto3.client('s3')
    
    # Configure the upload to use multipart
    config = TransferConfig(
        multipart_threshold=1024 * 25,  # 25MB
        max_concurrency=10,
        multipart_chunksize=1024 * 25,  # 25MB
        use_threads=True
    )
    
    # Get file size
    file_size = os.stat(local_file_path).st_size
    
    # Callback to track upload progress
    def upload_progress(bytes_transferred):
        print(f"Uploaded {bytes_transferred} bytes out of {file_size}")
    
    # Upload the file
    s3.upload_file(
        local_file_path, 
        bucket_name, 
        s3_file_path,
        Config=config,
        Callback=upload_progress,
        ExtraArgs={
            'ContentType': 'video/mp4'  # Adjust this based on your video format
        }
    )
    
    print(f"Video upload complete: {s3_uri}")

@dataclass 
class Config:
    # models can optionally be passed in directly
    model = None
    deployment_config = None
    
    # interrogator settings
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class AnimateAnyone():
    def __init__(self, config: Config, properties):
        self.history = None
        
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.load_model(properties)

        # interrogator settings
        device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, properties):
        
        deployment_config = OmegaConf.create(self.config.deployment_config)
        
        if deployment_config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        
        if self.config.model is None and deployment_config.model_name:
            model_dir = deployment_config.model_name
            if "model_id" in properties:
                model_dir = properties["model_id"]
                if any(os.listdir(model_dir)):
                    files_in_folder = os.listdir(model_dir)
                    print('model path files:')
                    for file in files_in_folder:
                        print(file)
                else:
                    raise ValueError('Please make sure the model artifacts are uploaded to s3')
            
            vae = AutoencoderKL.from_pretrained(
                f'{model_dir}/{deployment_config.pretrained_vae_path}',
            ).to("cuda", dtype=weight_dtype)

            reference_unet = UNet2DConditionModel.from_pretrained(
                f'{model_dir}/{deployment_config.pretrained_base_model_path}',
                subfolder="unet",
            ).to(dtype=weight_dtype, device="cuda")
            
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                f'{model_dir}/{deployment_config.pretrained_base_model_path}',
                f'{model_dir}/{deployment_config.motion_module_path}',
                subfolder="unet",
                unet_additional_kwargs=deployment_config.infer_config.unet_additional_kwargs,
            ).to(dtype=weight_dtype, device="cuda")

            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=weight_dtype, device="cuda"
            )

            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                f'{model_dir}/{deployment_config.image_encoder_path}'
            ).to(dtype=weight_dtype, device="cuda")

            sched_kwargs = deployment_config.infer_config.noise_scheduler_kwargs
            scheduler = DDIMScheduler(**sched_kwargs)

            # load pretrained weights
            denoising_unet.load_state_dict(
                torch.load(f'{model_dir}/{deployment_config.denoising_unet_path}', map_location="cpu"),
                strict=False,
            )
            reference_unet.load_state_dict(
                torch.load(f'{model_dir}/{deployment_config.reference_unet_path}', map_location="cpu"),
            )
            pose_guider.load_state_dict(
                torch.load(f'{model_dir}/{deployment_config.pose_guider_path}', map_location="cpu"),
            )

            pipe = Pose2VideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            self.model = pipe.to("cuda", dtype=weight_dtype)
        else:
            self.model = self.config.model
            
    def get_model(self):
        return self.model

with open('./deployment_config.json', 'rb') as openfile:
    deployment_config = json.load(openfile)

config = None
_service = None

def handle(inputs: Input) -> Optional[Output]:
    global config, _service
    if not _service:
        config = Config()
        config.deployment_config = deployment_config
        _service = AnimateAnyone(config, inputs.get_properties())
    
    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    tmp_dir = "/tmp"
    
    # Download pose sequence and reference image
    pose_video_path = f'{tmp_dir}/pose-sequence.mp4'
    ref_image_path = f'{tmp_dir}/reference.png'
    download_from_s3_uri(data['pose_seq_s3uri'], pose_video_path)
    download_from_s3_uri(data['ref_s3_path'], ref_image_path)
    height = data['height']
    width = data['width']
    length = data['length']
    steps = data['steps']
    cfg = data['cfg']
    fps = data['fps']
    seed = data['seed']
    output_s3uri = data['output_s3uri']
    
    generator = torch.manual_seed(seed)
    
    # Inference
    ref_name = Path(ref_image_path).stem
    pose_name = Path(pose_video_path).stem.replace("_kps", "")
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_list = []
    pose_tensor_list = []
    pose_images = read_frames(pose_video_path)
    src_fps = get_fps(pose_video_path)
    print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    
    for pose_image_pil in pose_images[: length]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)

    ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
        0
    )  # (1, c, 1, h, w)
    ref_image_tensor = repeat(
        ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=length
    )

    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)
    
    try:
        video = _service.get_model()(
            ref_image_pil,
            pose_list,
            width,
            height,
            length,
            steps,
            cfg,
            generator=generator,
        ).videos

        save_path = f"{tmp_dir}/generated_video.mp4"
        save_videos_grid(
            video,
            save_path,
            n_rows=1,
            fps=src_fps if fps < 1 else fps,
        )
        
        upload_video_to_s3(save_path, output_s3uri)
        return Output().add([output_s3uri])
    except:
        return Output().add(["Error occurred during inference"])