# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import cv2
import hashlib
from urllib.parse import urlparse, parse_qs
import urllib
import urllib.request
from urllib.request import Request


import torch
import base64
import tyro
from decord import VideoReader
from decord import cpu, gpu
from cog import BasePredictor, Input, Path
import uuid
import sys
import pathlib
from comfyui_utils import comfyui_generate_face_expression
from pathlib import Path as pathlibPath
home = pathlibPath.home()

print(f"current dir: {pathlib.Path().resolve()}")
sys.path.append(f'{home}/LivePortrait')
sys.path.append(f'{home}/ComfyUI')
from nodes import LoadImage
from importlib import import_module

from cog_server_utils import b2_upload_file_and_get_url, get_uuid

python_code = import_module('custom_nodes.ComfyUI-AdvancedLivePortrait.nodes')




its = python_code.ExpressionEditor.INPUT_TYPES()
expression_editor = python_code.ExpressionEditor()
expression_editor_kwargs = {}
for p,v in its['required'].items():
    if isinstance(v[0], list):
        expression_editor_kwargs[p] = v[0][0]
    else:
        expression_editor_kwargs[p] = v[1]['default']




from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

from PIL import Image, ImageSequence
import mimetypes
import requests   
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff




def download_image(url, local_path):

    headers = {'User-Agent': 'curl/7.65.2', 'Accept': '*/*'}

    r = requests.get(url, allow_redirects=True, headers=headers)
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)


def get_uuid():
    return str(uuid.uuid4())

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

DOWNLOAD_LOCAL_DIR = f"/tmp/local_download_uploaded/"
GEND_FACE_LOCAL_DIR = f"{home}/GEND_FACES/"

class Predictor(BasePredictor):
    live_portrait_pipeline = None
    live_portrait_args = None

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        try:
            tyro.extras.set_accent_color("bright_cyan")
            self.live_portrait_args = tyro.cli(ArgumentConfig, args=[])
            # specify configs for inference
            inference_cfg = partial_fields(InferenceConfig, self.live_portrait_args.__dict__)  # use attribute of args to initial InferenceConfig
            crop_cfg = partial_fields(CropConfig, self.live_portrait_args.__dict__)  # use attribute of args to initial CropConfig

            print(f"creating LivePortraitPipeline with cfg: {inference_cfg}, crop_cfg:{crop_cfg}")
            self.live_portrait_pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg
            )
            print(f"done setup")
        except Exception as ex:
            print(f"error: {ex}")
            raise ex
        
    def process_base64_or_url(self, input_str : str, local_download_dir : str):
        """base64 enocded image looks like: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."""

        if not os.path.exists(local_download_dir):
            Path(local_download_dir).mkdir(parents=True, exist_ok=True)

        if input_str is not None and input_str.startswith("data:image"):
            # this is a base64 encoded image
            meta_data, image_data = input_str.split(",")
            image_file_id = get_uuid()
            # detect extension from meta_data
            extension = meta_data.split('/')[1].split(';')[0]
            image_path = f"{local_download_dir}/{image_file_id}.{extension}"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            return image_path
        elif input_str is not None and input_str.startswith("http"):
            parsed_image_url = urlparse(input_str)
            parsed = parse_qs(parsed_image_url.query)
            if 'fileId' in parsed:
                image_file_id = parsed['fileId'][0]
            else:
                image_file_id = get_uuid()
            
            req = Request(input_str)
            req.add_header('User-Agent', 'curl/7.65.2')
            req.add_header('Accept', '*/*')

            # Get extension from metadata
            print(f"getting metadata from {input_str}")
            meta_data = urllib.request.urlopen(req).info()
            extension = meta_data.get_content_subtype()
            print(f"extension: {extension}, meta_data: {meta_data}")
            
            download_image_filename = image_file_id + f".{extension}"
            image_path = f"{local_download_dir}/{download_image_filename}"  
            if not os.path.exists(image_path):
                Path(local_download_dir).mkdir(parents=True, exist_ok=True)
                print(f"downloading image from {input_str} to {image_path}")
                # urllib.request.urlretrieve(input_str, image_path)
                download_image(input_str, image_path)
                print(f"downloaded image as: {image_path}")
            return image_path
        else:
            raise ValueError(f"Invalid input: {input_str}")
        
    def edit_expression(
            self,
            source_image_url, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, 
            pupil_x, pupil_y, aaa, eee, woo, smile, src_ratio):
        try:
            # update expression editor kwargs
            for k,v in locals().items():
                if k in expression_editor_kwargs:
                    expression_editor_kwargs[k] = v

            local_image_path = self.process_base64_or_url(source_image_url, DOWNLOAD_LOCAL_DIR)
            im = LoadImage().load_image(local_image_path)
            expression_editor_kwargs['src_image'] = im[0]

            print(f"expression_editor_kwargs: {expression_editor_kwargs}")

            out = expression_editor.run(**expression_editor_kwargs)

            if not os.path.exists(f"{home}/ComfyUI/temp/"+ out['ui']['images'][0]['filename']):
                raise ValueError(f"Failed to edit")
        except Exception as ex:
            print(f"error: {ex}")
            raise ex
        b2_url = b2_upload_file_and_get_url(f"{home}/ComfyUI/temp/" + out['ui']['images'][0]['filename'], None)
        print(f"uploaded to b2: {b2_url}")
        return b2_url


    def predict(
        self,
        image: Path = Input(description="input source image", default=None),
        source_image: str = Input(description="input source image base64 or url", default=None),
        driving_video_url: str = Input(description="driving_video_b2_url", default=None),
        driving_image: str = Input(description="driving_image_b2_url or base64", default=None),
        frame_ids: list[int] = Input(description="frame ids", default=[361]),
        driving_video_ext: str = Input(description="driving video ext can be .pkl or .mp4", default='.pkl'),
        return_video: bool = Input(description="if return generated video", default=False),
        return_template: bool = Input(description="if return generated video template", default=False),
        flag_relative_motion: bool = Input(description="if use relative motion", default=True),
        # expression editor parameters:
        is_expression_editor: bool = Input(description="if use expression editor", default=False),
        source_image_url: str = Input(description="input source image", default=None),
        rotate_pitch: float = Input(description="rotate pitch", default=expression_editor_kwargs['rotate_pitch']),
        rotate_yaw: float = Input(description="rotate yaw", default=expression_editor_kwargs['rotate_yaw']),
        rotate_roll: float = Input(description="rotate roll", default=expression_editor_kwargs['rotate_roll']),
        blink: float = Input(description="blink", default=expression_editor_kwargs['blink']),
        eyebrow: float = Input(description="eyebrow", default=expression_editor_kwargs['eyebrow']),
        wink: float = Input(description="wink", default=expression_editor_kwargs['wink']),
        pupil_x: float = Input(description="pupil x", default=expression_editor_kwargs['pupil_x']),
        pupil_y: float = Input(description="pupil y", default=expression_editor_kwargs['pupil_y']),
        aaa: float = Input(description="aaa", default=expression_editor_kwargs['aaa']),
        eee: float = Input(description="eee", default=expression_editor_kwargs['eee']),
        woo: float = Input(description="woo", default=expression_editor_kwargs['woo']),
        smile: float = Input(description="smile", default=expression_editor_kwargs['smile']),
        src_ratio: float = Input(description="src ratio", default=expression_editor_kwargs['src_ratio']),
        # force comfyui:
        force_comfyui: bool = Input(description="force comfyui", default=False),
        sample_parts: str = Input(description="only generate expression, only rotation or all", default='All'),
    ) -> list[str]:
        
        print(f"force_comfyui: {force_comfyui}")

        if force_comfyui:
            ret = comfyui_generate_face_expression(locals())
            print(f"comfyui ret: {ret}")
            return [ret['output']]
        
        print(f"is_expression_editor: {is_expression_editor}")
        
        if is_expression_editor:
            print(f"expression editing...")
            ret = self.edit_expression(str(source_image_url), rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile, src_ratio)
            return [ret]
        """Run a list of gend faces"""

        """base64 enocded image looks like: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."""
        self.live_portrait_args.flag_relative_motion = flag_relative_motion
        try:

            # 1. setup driving image or video download if it's url or decode base64
            # if passed driving_image, use it first, otherwise use driving_video_url
            if driving_image is not None:
                self.live_portrait_args.driving = self.process_base64_or_url(driving_image, DOWNLOAD_LOCAL_DIR)

            # 2 setup source image
            # if passed image, use it first, otherwise use source_image
            if image is not None:   
                self.live_portrait_args.source = str(image)
            elif source_image is not None and (source_image.startswith("data:image") or source_image.startswith("http")):
                self.live_portrait_args.source = self.process_base64_or_url(source_image, DOWNLOAD_LOCAL_DIR)
            else:
                raise ValueError(f"Invalid input: {source_image}")

            assert os.path.exists(self.live_portrait_args.source)
            assert os.path.exists(self.live_portrait_args.driving)
            print(f"running live portrait with source image: {self.live_portrait_args.source} driving video: {self.live_portrait_args.driving}")


            # 3 run
            wfp, wfp_concat = self.live_portrait_pipeline.execute(self.live_portrait_args)
            # https://github.com/KwaiVGI/LivePortrait/blob/main/src/live_portrait_pipeline.py: 415
            gend_video_file_path = wfp #f"/root/LivePortrait/animations/{fn_img}--{fn_video}.mp4"

            # 4 fetch frame id image
            assert os.path.exists(gend_video_file_path) and Path(gend_video_file_path).stat().st_size > 0

            to_ret = []
            if driving_image is not None:
                # it's not a video, don't need to extract frames
                b2_url = b2_upload_file_and_get_url(gend_video_file_path, None)
                to_ret.append(b2_url)
            else:
                vr = VideoReader(gend_video_file_path, ctx=cpu(0))


                for frame_id in frame_ids:

                    face_img = Image.fromarray(vr[frame_id].asnumpy())

                    face_img_uuid = get_uuid()

                    Path(GEND_FACE_LOCAL_DIR).mkdir(parents=True, exist_ok=True)

                    face_img.save(f"{GEND_FACE_LOCAL_DIR}/{face_img_uuid}.png")

                    to_ret.append(Path(f"{GEND_FACE_LOCAL_DIR}/{face_img_uuid}.png"))

            if driving_image is not None and return_video:
                b2_url = b2_upload_file_and_get_url(gend_video_file_path, None)
                to_ret.append(b2_url)
            if driving_image is not None and return_template and driving_video_ext == ".mp4":
                b2_url = b2_upload_file_and_get_url(Path(driving_video_path.replace(".mp4", ".pkl")), None)
                to_ret.append(b2_url)

            return to_ret
        except Exception as ex:
            print(f"error: {ex}")
            raise ex
