import os
import numpy as np
import cv2
import requests
import onnxruntime


def download_model(model_name):
    model_info = upscale_models[model_name]
    model_url = model_info['url']
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "upscalers")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.onnx")
    if not os.path.exists(model_path):
        print(f"Downloading model {model_name}")
        r = requests.get(model_url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    return model_path


upscale_models = {
    'clear_reality_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/clear_reality_x4.onnx',
        'size': (128, 8, 4),
        'scale': 4
    },
    'lsdir_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/lsdir_x4.onnx',
        'size': (128, 8, 4),
        'scale': 4
    },
    'nomos8k_sc_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/nomos8k_sc_x4.onnx',
        'size': (128, 8, 4),
        'scale': 4
    },
    'real_esrgan_x2': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x2.onnx',
        'size': (256, 16, 8),
        'scale': 2
    },
    'real_esrgan_x2_fp16': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x2_fp16.onnx',
        'size': (256, 16, 8),
        'scale': 2
    },
    'real_esrgan_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x4.onnx',
        'size': (256, 16, 8),
        'scale': 4
    },
    'real_esrgan_x4_fp16': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_esrgan_x4_fp16.onnx',
        'size': (256, 16, 8),
        'scale': 4
    },
    'real_esrgan_x8': {
        'url': 'https://huggingface.co/facefusion/models/resolve/main/real_esrgan_x8.onnx',
        'size': (128, 8, 2),
        'scale': 8
    },
    'real_esrgan_x8_fp16': {
        'url': 'https://huggingface.co/facefusion/models/resolve/main/real_esrgan_x8_fp16.onnx',
        'size': (128, 8, 2),
        'scale': 8
    },
    'real_hatgan_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/real_hatgan_x4.onnx',
        'size': (256, 16, 8),
        'scale': 4
    },
    'span_kendata_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/span_kendata_x4.onnx',
        'size': (128, 8, 4),
        'scale': 4
    },
    'ultra_sharp_x4': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/ultra_sharp_x4.onnx',
        'size': (128, 8, 4),
        'scale': 4
    }
}


class SUPiRUpscaler:
    def __init__(self, model_name: str = None):
        self.model = None
        self.model_name = model_name
        self.ort_session = None
        if model_name:
            self.load_model(model_name)

    def load_model(self, model_name: str):
        model_path = download_model(model_name)
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.model_name = model_name

    def prepare_tile_frame(self, vision_tile_frame: np.ndarray) -> np.ndarray:
        vision_tile_frame = np.expand_dims(vision_tile_frame[:, :, ::-1], axis=0)
        vision_tile_frame = vision_tile_frame.transpose(0, 3, 1, 2)
        vision_tile_frame = vision_tile_frame.astype(np.float32) / 255
        return vision_tile_frame

    def normalize_tile_frame(self, vision_tile_frame: np.ndarray) -> np.ndarray:
        vision_tile_frame = vision_tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        vision_tile_frame = vision_tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return vision_tile_frame

    def process(self, image: np.ndarray) -> np.ndarray:
        size = upscale_models[self.model_name]['size']
        scale = upscale_models[self.model_name]['scale']
        temp_height, temp_width = image.shape[:2]
        input_image = self.prepare_tile_frame(image)

        ort_inputs = {self.ort_session.get_inputs()[0].name: input_image}
        ort_outs = self.ort_session.run(None, ort_inputs)

        tile_vision_frame = ort_outs[0]
        tile_vision_frame = self.normalize_tile_frame(tile_vision_frame)

        image = cv2.resize(image, (tile_vision_frame.shape[1], tile_vision_frame.shape[0]))
        image = cv2.addWeighted(image, 0.8, tile_vision_frame, 0.2, 0)
        return image

