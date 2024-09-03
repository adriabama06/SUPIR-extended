import os
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import requests
from huggingface_hub.constants import (
    DEFAULT_ETAG_TIMEOUT,
    DEFAULT_REVISION,
    REPO_TYPES,
)
from huggingface_hub.file_download import REGEX_COMMIT_HASH, hf_hub_download, repo_folder_name
from huggingface_hub.hf_api import DatasetInfo, HfApi, ModelInfo, SpaceInfo
from huggingface_hub.utils import (
    GatedRepoError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    filter_repo_objects,
    logging,
    validate_hf_hub_args,
)
from huggingface_hub.utils import tqdm as hf_tqdm
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map

logger = logging.get_logger(__name__)


@validate_hf_hub_args
def rd_download(
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        local_dir: Union[str, Path, None] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Optional[Union[Dict, str]] = None,
        proxies: Optional[Dict] = None,
        etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
        force_download: bool = False,
        token: Optional[Union[bool, str]] = None,
        local_files_only: bool = False,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        max_workers: int = 8,
        tqdm_class: Optional[base_tqdm] = None,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        # Deprecated args
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        resume_download: Optional[bool] = None,
) -> str:
    """Download repo files directly to the specified directory or temporarily cache in /tmp."""

    if revision is None:
        revision = DEFAULT_REVISION

    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")

    temp_cache_dir = "/tmp/huggingface_hub_cache"
    storage_folder = os.path.join(temp_cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    repo_info: Union[ModelInfo, DatasetInfo, SpaceInfo, None] = None
    api_call_error: Optional[Exception] = None
    if not local_files_only:
        try:
            api = HfApi(
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
                endpoint=endpoint,
                headers=headers,
            )
            repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision, token=token)
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            raise
        except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OfflineModeIsEnabled,
        ) as error:
            api_call_error = error
            pass
        except RevisionNotFoundError:
            raise
        except requests.HTTPError as error:
            api_call_error = error
            pass

    if repo_info is None:
        commit_hash = None
        if REGEX_COMMIT_HASH.match(revision):
            commit_hash = revision
        else:
            ref_path = os.path.join(storage_folder, "refs", revision)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    commit_hash = f.read()

        if commit_hash is not None:
            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if os.path.exists(snapshot_folder):
                return snapshot_folder

        if local_files_only:
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass "
                "'local_files_only=False' as input."
            )
        elif isinstance(api_call_error, OfflineModeIsEnabled):
            raise LocalEntryNotFoundError(
                "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and "
                "outgoing traffic has been disabled. To enable repo look-ups and downloads online, set "
                "'HF_HUB_OFFLINE=0' as environment variable."
            ) from api_call_error
        elif isinstance(api_call_error, RepositoryNotFoundError) or isinstance(api_call_error, GatedRepoError):
            raise api_call_error
        else:
            raise LocalEntryNotFoundError(
                "An error happened while trying to locate the files on the Hub and we cannot find the appropriate"
                " snapshot folder for the specified revision on the local disk. Please check your internet connection"
                " and try again."
            ) from api_call_error

    assert repo_info.sha is not None, "Repo info returned from server must have a revision sha."
    assert repo_info.siblings is not None, "Repo info returned from server must have a siblings list."
    filtered_repo_files = list(
        filter_repo_objects(
            items=[f.rfilename for f in repo_info.siblings],
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )
    commit_hash = repo_info.sha
    snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
    if revision != commit_hash:
        ref_path = os.path.join(storage_folder, "refs", revision)
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

    def _inner_hf_hub_download(repo_file: str):
        return hf_hub_download(
            repo_id,
            filename=repo_file,
            repo_type=repo_type,
            revision=commit_hash,
            endpoint=endpoint,
            cache_dir=temp_cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            proxies=proxies,
            etag_timeout=etag_timeout,
            resume_download=resume_download,
            force_download=force_download,
            token=token,
            headers=headers,
        )

    if max_workers > 1:
        thread_map(
            _inner_hf_hub_download,
            filtered_repo_files,
            desc=f"Fetching {len(filtered_repo_files)} files",
            max_workers=max_workers,
            tqdm_class=tqdm_class or hf_tqdm,
        )
    else:
        for file in filtered_repo_files:
            _inner_hf_hub_download(file)

    if local_dir is not None:
        local_dir_path = str(os.path.realpath(local_dir))
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        return local_dir_path
    else:
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        return snapshot_folder


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
        rd_download(repo, local_dir=model_path, allow_patterns=allow_patterns)


if __name__ == '__main__':
    download_models()
