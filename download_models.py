import os

from huggingface_hub import snapshot_download

# Set the environment variable HF_HUB_ENABLE_HF_TRANSFER=1
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'


def download_models(download_lightning: bool = True):
    # Repo, files, folder
    repos = [
        ('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', 'open_clip_pytorch_model.bin', None),
        #('ashleykleynhans/SUPIR', ['SUPIR-v0F.ckpt', 'SUPIR-v0Q.ckpt'], None),
        ('Kijai/SUPIR_pruned', ['SUPIR-v0Q_fp16.safetensors', 'SUPIR-v0F_fp16.safetensors'], None),
        ('vikhyatk/moondream2', None, 'moondream2'),
        ('openai/clip-vit-large-patch14-336', None, None),
        ('openai/clip-vit-large-patch14', None, None)
    ]
    if download_lightning:
        repos.append(
            ('RunDiffusion/Juggernaut-XL-Lightning', 'Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors',
             'checkpoints'))
    else:
        repos.append(('RunDiffusion/Juggernaut-X-v10', 'Juggernaut-X-RunDiffusion-NSFW.safetensors', 'checkpoints'))

    model_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

    # If models/V0F.ckpt or models/V0Q.ckpt exist, rename them to 'SUPIR-v0F.ckpt' and 'SUPIR-v0Q.ckpt'
    if os.path.exists(os.path.join(model_dir, 'V0F.ckpt')):
        os.rename(os.path.join(model_dir, 'V0F.ckpt'), os.path.join(model_dir, 'SUPIR-v0F.ckpt'))

    if os.path.exists(os.path.join(model_dir, 'V0Q.ckpt')):
        os.rename(os.path.join(model_dir, 'V0Q.ckpt'), os.path.join(model_dir, 'SUPIR-v0Q.ckpt'))

    for repo, model_file, model_folder in repos:
        print(f"Checking for {repo} model...")
        if model_file is None:
            model_folder = repo.split('/')[1]
            model_path = os.path.join(model_dir, model_folder)
            allow_patterns = None
        else:
            if model_folder is None:
                model_path = model_dir
            else:
                model_path = os.path.join(model_dir, model_folder)
            if isinstance(model_file, list):
                allow_patterns = model_file
            else:
                allow_patterns = [model_file]
        snapshot_download(repo, local_dir=model_path, allow_patterns=allow_patterns, cache_dir="/tmp/SUPIR")


if __name__ == '__main__':
    download_models()
