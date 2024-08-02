import datetime
import json
import os
import platform
import re
import subprocess
from typing import List, Dict, Any

import cv2
import filetype
import gradio as gr
import torch
from PIL import Image
from ffmpeg_progress_yield import FfmpegProgress
from tqdm import tqdm

from SUPIR.perf_timer import PerfTimer
from SUPIR.utils.status_container import MediaData


def is_video(video_path: str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def detect_hardware_acceleration() -> (str, str, str):
    hw_accel_methods = [
        {'name': 'cuda', 'encoder': 'h264_nvenc', 'decoder': 'h264_cuvid', 'regex': re.compile(r'\bh264_nvenc\b')},
        {'name': 'qsv', 'encoder': 'h264_qsv', 'decoder': 'h264_qsv', 'regex': re.compile(r'\bh264_qsv\b')},
        {'name': 'vaapi', 'encoder': 'h264_vaapi', 'decoder': 'h264_vaapi', 'regex': re.compile(r'\bh264_vaapi\b')},
        # Add more methods here as needed, following the same structure
    ]

    ffmpeg_output = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True).stdout

    for method in hw_accel_methods:
        if method['regex'].search(ffmpeg_output):
            # Hardware acceleration method found
            return method['name'], method['decoder'], method['encoder']

    # No supported hardware acceleration found
    return '', '', ''


def extract_video(video_path: str, output_path: str, quality: int = 100, format: str = 'png', video_start=None,
                  video_end=None) -> (
        bool, Dict[str, str]):
    video_params = get_video_params(video_path)
    temp_frame_compression = 31 - (quality * 0.31)
    trim_frame_start = video_start
    trim_frame_end = video_end
    target_path = output_path
    printt(f"Extracting frames to: {target_path}, {format}")
    temp_frames_pattern = os.path.join(target_path, '%04d.' + format)
    commands = ['-hwaccel', 'auto', '-i', video_path, '-q:v', str(temp_frame_compression), '-pix_fmt', 'rgb24']
    resolution = f"{video_params['width']}x{video_params['height']}"
    video_fps = video_params['framerate']
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend(['-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(
            trim_frame_end) + ',scale=' + resolution + ',fps=' + str(video_fps)])
    elif trim_frame_start is not None:
        commands.extend(
            ['-vf', 'trim=start_frame=' + str(trim_frame_start) + ',scale=' + resolution + ',fps=' + str(video_fps)])
    elif trim_frame_end is not None:
        commands.extend(['-vf',
                         'trim=end_frame=' + str(trim_frame_end) + ',scale=' + resolution + ',fps=' + str(
                             video_fps)])
    else:
        commands.extend(['-vf', 'scale=' + resolution + ',fps=' + str(video_fps)])
    commands.extend(['-vsync', '0', temp_frames_pattern])
    printt(f"Extracting frames from video: '{' '.join(commands)}'")
    video_params['start_frame'] = trim_frame_start
    video_params['end_frame'] = trim_frame_end
    return run_ffmpeg_progress(commands), video_params


def get_video_params(video_path: str) -> Dict[str, str]:
    # Command to get video dimensions, codec, frame rate, duration, and frame count
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,codec_name,nb_read_frames',
           '-show_entries', 'format=duration', '-count_frames', '-of', 'json', video_path]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Parse ffprobe output to json
        info = json.loads(result.stdout)
        # Extract video stream information
        stream = info['streams'][0]  # Assuming the first stream is the video
        # Extract format (container) information, including duration
        format_info = info['format']

        # Calculate framerate as float
        read_frames = int(stream['nb_read_frames'])
        duration = float(format_info['duration'])
        framerate = read_frames / duration if duration > 0 else float(stream['r_frame_rate'])

        return {
            'width': stream['width'],
            'height': stream['height'],
            'framerate': framerate,
            'fps': framerate,
            'codec': stream['codec_name'],
            'duration_seconds': duration,
            'frames': read_frames  # Number of frames (might be N/A if not available)
        }
    except Exception as e:
        print(f"Error extracting video parameters: {e}")
        return {}


def compile_video(src_video: str, extracted_path: str, output_path: str, video_params: Dict[str, str], quality: int = 1,
                  file_type: str = 'mp4', video_start=None, video_end=None) -> bool:
    # if quality is a string, just make it 1
    if isinstance(quality, str):
        quality = 1.0
    output_path_with_type = f"{output_path}.{file_type}"
    if os.path.exists(output_path_with_type):
        existing_idx = 1
        while os.path.exists(f"{output_path}_{existing_idx}.{file_type}"):
            existing_idx += 1
        output_path_with_type = f"{output_path}_{existing_idx}.{file_type}"

    temp_frames_pattern = os.path.join(extracted_path, '%04d.png')
    video_fps = video_params['framerate']
    output_video_encoder = 'libx264'
    commands = ['-hwaccel', 'auto', '-r', str(video_fps), '-i', temp_frames_pattern, '-c:v',
                output_video_encoder]
    if output_video_encoder in ['libx264', 'libx265', 'h264_nvenc', 'hevc_nvenc']:
        output_video_compression = round(51 - (quality * 0.51))
        if not "nvenc" in output_video_encoder:
            commands.extend(['-crf', str(output_video_compression), '-preset', 'veryfast'])
    if output_video_encoder in ['libvpx-vp9']:
        output_video_compression = round(63 - (quality * 0.63))
        commands.extend(['-crf', str(output_video_compression)])
    commands.extend(['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', output_path_with_type])
    printt(f"Merging frames to video: '{' '.join(commands)}'")
    if run_ffmpeg_progress(commands):
        image_data = MediaData(output_path_with_type, 'video')
        if restore_audio(src_video, output_path_with_type, video_fps, video_start, video_end):
            printt(f"Audio restored to video successfully: {output_path_with_type}")
        else:
            printt(f"Audio restoration failed: {output_path_with_type}")
        image_data.outputs = [output_path_with_type]
        return image_data
    return False


def run_ffmpeg_progress(args: List[str], progress=gr.Progress()):
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    commands.extend(args)
    print(f"Executing ffmpeg: '{' '.join(commands)}'")
    try:
        ff = FfmpegProgress(commands)
        last_progress = 0  # Keep track of the last progress value
        with tqdm(total=100, position=1, desc="Processing") as pbar:
            for p in ff.run_command_with_progress():
                increment = p - last_progress  # Calculate the increment since the last update
                pbar.update(increment)  # Update tqdm bar with the increment
                pbar.set_postfix(progress=p)
                progress(p / 100, "Extracting frames")  # Update gr.Progress with the normalized progress value
                last_progress = p  # Update the last progress value
        return True
    except Exception as e:
        print(f"Exception in run_ffmpeg_progress: {e}")
        return False


def get_video_frame(video_path: str, frame_number: int = 0):
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            frame_total = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
            has_vision_frame, vision_frame = video_capture.read()
            video_capture.release()
            if has_vision_frame:
                vision_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)
                return vision_frame
    return None


last_time = None
ui_args = None
timer = None


def printt(msg, progress=gr.Progress(), reset: bool = False):
    global ui_args, last_time, timer
    graph = None
    if ui_args is not None and ui_args.debug:
        if timer is None:
            timer = PerfTimer(print_log=True)
        if reset:
            graph = timer.make_graph()
            timer.reset()
        if not timer.print_log:
            timer.print_log = True
        timer.record(msg)
    else:
        print(msg)
    if graph:
        return graph


def restore_audio(src_video, target_video, video_fps, frame_start, frame_end) -> bool:
    output_video_path = os.path.splitext(target_video)[0] + "_audio_restored.mp4"
    commands = ['ffmpeg', '-hwaccel', 'auto']
    commands.extend(['-i', target_video])
    commands.extend(['-i', src_video])

    # Applying the frame cut if specified
    if frame_start is not None:
        start_time = frame_start / video_fps
        commands.extend(['-ss', str(start_time)])
    if frame_end is not None:
        end_time = frame_end / video_fps
        commands.extend(['-to', str(end_time)])

    # Copy video from target_video and audio from src_video, map them accordingly
    commands.extend(['-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-shortest', output_video_path])

    # Print command for debugging
    print(f"Executing FFmpeg command: {' '.join(commands)}")

    # Execute FFmpeg command
    try:
        subprocess.run(commands, check=True)
        print(f"Audio restored to video successfully: {output_video_path}")
        # Delete the original target_video
        os.remove(target_video)
        # Rename the restored video to the original target_video name
        os.rename(output_video_path, target_video)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restoring audio: {e}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        return False


def refresh_styles_click():
    new_style_list = list_styles()
    style_list = list(new_style_list.keys())
    return gr.update(choices=style_list)


def list_styles():
    styles_path = os.path.join(os.path.dirname(__file__), 'styles')
    output = {}
    style_files = []
    llava_prompt = default_llm_prompt
    for root, dirs, files in os.walk(styles_path):
        for file in files:
            if file.endswith('.csv'):
                style_files.append(os.path.join(root, file))
    for style_file in style_files:
        with open(style_file, 'r') as f:
            lines = f.readlines()
            # Parse lines, skipping the first line
            for line in lines[1:]:
                line = line.strip()
                if len(line) > 0:
                    name = line.split(',')[0]
                    cap_line = line.replace(name + ',', '')
                    captions = cap_line.split('","')
                    if len(captions) >= 2:
                        positive_prompt = captions[0].replace('"', '')
                        negative_prompt = captions[1].replace('"', '')
                        if "{prompt}" in positive_prompt:
                            positive_prompt = positive_prompt.replace("{prompt}", "")

                        if "{prompt}" in negative_prompt:
                            negative_prompt = negative_prompt.replace("{prompt}", "")

                        if len(captions) == 3:
                            llava_prompt = captions[2].replace('"', "")

                        output[name] = (positive_prompt, negative_prompt, llava_prompt)

    return output


def select_style(style_name, current_prompt=None, values=False):
    style_list = list_styles()

    if style_name in style_list.keys():
        style_pos, style_neg, style_llava = style_list[style_name]
        if values:
            return style_pos, style_neg, style_llava
        return gr.update(value=style_pos), gr.update(value=style_neg), gr.update(value=style_llava)
    if values:
        return "", "", ""
    return gr.update(value=""), gr.update(value=""), gr.update(value="")


def open_folder():
    from gradio_demo import args
    open_folder_path = os.path.abspath(args.outputs_folder)
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


def set_info_attributes(elements_to_set: Dict[str, Any]):
    output = {}
    for key, value in elements_to_set.items():
        if not getattr(value, 'elem_id', None):
            setattr(value, 'elem_id', key)
        classes = getattr(value, 'elem_classes', None)
        if isinstance(classes, list):
            if "info-btn" not in classes:
                classes.append("info-button")
                setattr(value, 'elem_classes', classes)
        output[key] = value
    return output


def list_models(model_dir: str, ckpt: str):
    output = []
    if os.path.exists(model_dir):
        output = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if
                  f.endswith('.safetensors') or f.endswith('.ckpt')]
    else:
        local_model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        if os.path.exists(local_model_dir):
            output = [os.path.join(local_model_dir, f) for f in os.listdir(local_model_dir) if
                      f.endswith('.safetensors') or f.endswith('.ckpt')]
    if os.path.exists(ckpt) and ckpt not in output:
        output.append(ckpt)
    else:
        if os.path.exists(os.path.join(os.path.dirname(__file__), ckpt)):
            output.append(os.path.join(os.path.dirname(__file__), ckpt))
    # Sort the models
    output = [os.path.basename(f) for f in output]
    # Ensure the values are unique
    output = list(set(output))
    output.sort()
    return output


def get_ckpt_path(ckpt_path: str, model_dir: str):
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        if os.path.exists(model_dir):
            return os.path.join(model_dir, ckpt_path)
        local_model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        if os.path.exists(local_model_dir):
            return os.path.join(local_model_dir, ckpt_path)
    return None


def selected_model(ckpt, ckpt_dir):
    models = list_models(ckpt_dir, ckpt_dir)
    target_model = ckpt
    if os.path.basename(target_model) in models:
        return target_model
    else:
        if len(models) > 0:
            return models[0]
    return None


def to_gpu(elem_to_load, device):
    if elem_to_load is not None:
        elem_to_load = elem_to_load.to(device)
        if getattr(elem_to_load, 'move_to', None):
            elem_to_load.move_to(device)
        if getattr(elem_to_load, 'to', None):
            elem_to_load = elem_to_load.to(device)
        torch.cuda.set_device(device)
    return elem_to_load


slider_html = """
<div id="keyframeSlider" class="keyframe-slider">
  <div id="frameSlider"></div>

  <!-- Labels for start and end times -->
  <div class="labels">
    <span id="startTimeLabel">0:00:00</span>
    <span id="nowTimeLabel">0:00:30</span>
    <span id="endTimeLabel">0:01:00</span>
  </div>
</div>
"""
title_md = """
# **SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration**

1 Click Installer (auto download models as well) : https://www.patreon.com/posts/99176057

FFmpeg Install Tutorial : https://youtu.be/-NjNy7afOQ0 &emsp; [[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[How to play](https://github.com/Fanghua-Yu/SUPIR/blob/master/assets/DemoGuide.png)]
"""
claim_md = """
## **Terms of use**

By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. Please submit a feedback to us if you get any inappropriate answer! We will collect those to keep improving our models. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

## **License**
While the original readme for the project *says* it's non-commercial, it was *actually* released under the MIT license. That means that the project can be used for whatever you want.

And yes, it would certainly be nice if anything anybody stuck in a random readme were the ultimate gospel when it comes to licensing, unfortunately, that's just
not how the world works. MIT license means FREE FOR ANY PURPOSE, PERIOD.
The service is a research preview ~~intended for non-commercial use only~~, subject to the model [License](https://github.com/Fanghua-Yu/SUPIR#MIT-1-ov-file) of SUPIR.
"""
refresh_symbol = "\U000027F3"  # ⟳
dl_symbol = "\U00002B73"  # ⭳
fullscreen_symbol = "\U000026F6"  # ⛶
default_llm_prompt = "Describe this image and its style in a very detailed manner. The image is a realistic photography, not an art painting."


def update_model_settings(model_type, param_setting):
    """
    Returns a series of gr.updates with settings based on the model type.
    If 'model_type' contains 'lightning', it uses the settings for a 'lightning' SDXL model.
    Otherwise, it uses the settings for a normal SDXL model.
    s_cfg_Quality, spt_linear_CFG_Quality, s_cfg_Fidelity, spt_linear_CFG_Fidelity, edm_steps
    """
    # Default settings for a "lightning" SDXL model
    lightning_settings = {
        's_cfg_Quality': 2.0,
        'spt_linear_CFG_Quality': 2.0,
        's_cfg_Fidelity': 1.5,
        'spt_linear_CFG_Fidelity': 1.5,
        'edm_steps': 8
    }

    # Default settings for a normal SDXL model
    normal_settings = {
        's_cfg_Quality': 7.5,
        'spt_linear_CFG_Quality': 4.0,
        's_cfg_Fidelity': 4.0,
        'spt_linear_CFG_Fidelity': 1.0,
        'edm_steps': 50
    }

    # Choose the settings based on the model type
    settings = lightning_settings if 'Lightning' in model_type else normal_settings

    if param_setting == "Quality":
        s_cfg = settings['s_cfg_Quality']
        spt_linear_CFG = settings['spt_linear_CFG_Quality']
    else:
        s_cfg = settings['s_cfg_Fidelity']
        spt_linear_CFG = settings['spt_linear_CFG_Fidelity']

    return gr.update(value=s_cfg), gr.update(value=spt_linear_CFG), gr.update(value=settings['edm_steps'])


def read_image_metadata(image_path):
    if image_path is None:
        return
    # Check if the file exists
    if not os.path.exists(image_path):
        return "File does not exist."

    # Get the last modified date and format it
    last_modified_timestamp = os.path.getmtime(image_path)
    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')

    # Open the image and extract metadata
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1e6

        metadata_str = f"Last Modified Date: {last_modified_date}\nMegapixels: {megapixels:.2f}\n"

        # Extract metadata based on image format
        if img.format == 'JPEG':
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = Image.ExifTags.TAGS.get(tag, tag)
                    metadata_str += f"{tag_name}: {value}\n"
        else:
            metadata = img.info
            if metadata:
                for key, value in metadata.items():
                    metadata_str += f"{key}: {value}\n"
            else:
                metadata_str += "No additional metadata found."

    return metadata_str


def submit_feedback(evt_id, f_score, f_text):
    from gradio_demo import args
    if args.log_history:
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'r') as f:
            event_dict = eval(f.read())
        f.close()
        event_dict['feedback'] = {'score': f_score, 'text': f_text}
        with open(f'./history/{evt_id[:5]}/{evt_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        return 'Submit successfully, thank you for your comments!'
    else:
        return 'Submit failed, the server is not set to log history.'


def refresh_models_click():
    from gradio_demo import args
    new_model_list = list_models(args.ckpt_dir, args.ckpt)
    return gr.update(choices=new_model_list)
