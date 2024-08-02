from huggingface_hub import snapshot_download
import os

models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'models'))


def get_model(model_repo: str):
    if os.path.exists(model_repo):
        return model_repo
    model_name = model_repo.split('/')[-1]
    model_path = os.path.join(models_folder, model_name)
    if not os.path.exists(model_path):
        model_folder = model_repo.split('/')[1]
        model_repo = os.environ.get(model_repo, model_repo)
        if os.environ.get("HF_HUB_TOKEN"):
            token = os.environ.get("HF_HUB_TOKEN")
            snapshot_download(model_repo, local_dir=os.path.join(models_folder, model_folder), local_dir_use_symlinks=False, token=token)
        else:
            snapshot_download(model_repo, local_dir=os.path.join(models_folder, model_folder), local_dir_use_symlinks=False)
    return model_path
