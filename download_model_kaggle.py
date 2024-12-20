import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
import subprocess

# Set environment variable for faster HF downloads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Directory created: {path}')
    else:
        print(f'Directory already exists: {path}')

def download_file_aria2(url, folder_path, file_name=None):
    """Download a file using aria2c with 16 connections."""
    local_filename = file_name if file_name else url.split('/')[-1]
    local_filepath = os.path.join(folder_path, local_filename)

    # Check if file exists and verify its size
    if os.path.exists(local_filepath):
        print(f'File already exists: {local_filepath}')
        expected_size = get_remote_file_size(url)
        actual_size = os.path.getsize(local_filepath)
        if expected_size == actual_size:
            print(f'File is already downloaded and verified: {local_filepath}')
            return
        else:
            print(f'File size mismatch, redownloading: {local_filepath}')

    print(f'Downloading {url} to: {local_filepath}')
    
    # aria2c command with 16 connections
    command = [
        'aria2c',
        url,
        f'--dir={folder_path}',
        f'--out={local_filename}',
        '--max-connection-per-server=16',
        '--split=16',
        '--min-split-size=1M',
        '--continue=true',
        '--max-concurrent-downloads=16',
        '--file-allocation=none'
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f'Downloaded {local_filename} to {folder_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error downloading file: {e}')

def get_remote_file_size(url):
    """Get the size of a file at a remote URL."""
    with requests.head(url) as r:
        size = int(r.headers.get('content-length', 0))
    return size

# Define the folders and their corresponding file URLs with optional file names
folders_and_files = {
    os.path.join('models'): [
        ('https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin', None),
        ('https://huggingface.co/ashleykleynhans/SUPIR/resolve/main/SUPIR-v0F.ckpt', 'v0F.ckpt'),
        ('https://huggingface.co/ashleykleynhans/SUPIR/resolve/main/SUPIR-v0Q.ckpt', 'v0Q.ckpt')
    ]
}

# Main execution
for folder, files in folders_and_files.items():
    create_directory(folder)
    for file_url, file_name in files:
        download_file_aria2(file_url, folder, file_name)

llava_model = os.getenv('LLAVA_MODEL', 'liuhaotian/llava-v1.5-7b')
llava_clip_model = 'openai/clip-vit-large-patch14-336'
sdxl_clip_model = 'openai/clip-vit-large-patch14'

# For HuggingFace models, we'll continue using snapshot_download as it's more reliable for these repositories
print(f'Downloading LLaVA model: {llava_model}')
model_folder = llava_model.split('/')[1]
snapshot_download(llava_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)

print(f'Downloading LLaVA CLIP model: {llava_clip_model}')
model_folder = llava_clip_model.split('/')[1]
snapshot_download(llava_clip_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)

print(f'Downloading SDXL CLIP model: {sdxl_clip_model}')
model_folder = sdxl_clip_model.split('/')[1]
snapshot_download(sdxl_clip_model, local_dir=os.path.join("models", model_folder), local_dir_use_symlinks=False)
