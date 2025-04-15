def rename_meta_key(key):
    key_mapping = {
        "s_cfg": "Text Guidance Scale",
        "face_prompt":"Used Face Restore Prompt",
        "caption":"Used Final Prompt",
        "upscale":"Upscale Ratio",
        "s_stage2": "Stage2 Guidance Strength",
        "s_stage1": "Stage1 Guidance Strength",
        "spt_linear_CFG": "Linear CFG Start",
        "spt_linear_s_stage2": "Linear Stage2 Guidance Start",
        "top_p": "LLaVA Top P",
        "main_prompt": "User Provided Prompt",
        "temperature": "LLaVA Temperature",
        "face_resolution":"Face Options Text Guidance Scale",
        "a_prompt": "Default Positive Prompt",
        "n_prompt": "Default Negative Prompt",
        "ae_dtype": "Auto-Encoder Data Type",
        "diff_dtype": "Diffusion Data Type",
        "edm_steps": "Number Of Steps",
        "apply_bg": "Face Options BG restoration",
        "apply_face": "Face Options Face restoration",
        "apply_llava": "Auto Caption With LLaVA",
        "apply_supir": "Upscale With SUPIR",
        "ckpt_select": "Used Base Model",
        "ckpt_type": "Checkpoint Type",
        "max_megapixels": "max_megapixels",
        "max_resolution": "max_resolution"
    }
    return key_mapping.get(key, key)

def rename_meta_key_reverse(key):
    key_mapping = {
        "Text Guidance Scale": "s_cfg",
        "Used Face Restore Prompt": "face_prompt",
        "Used Final Prompt": "caption",
        "Upscale Ratio": "upscale",
        "Stage2 Guidance Strength": "s_stage2",
        "Stage1 Guidance Strength": "s_stage1",
        "Linear CFG Start": "spt_linear_CFG",
        "Linear Stage2 Guidance Start": "spt_linear_s_stage2",
        "LLaVA Top P": "top_p",
        "User Provided Prompt": "main_prompt",
        "LLaVA Temperature": "temperature",
        "Face Options Text Guidance Scale": "face_resolution",
        "Default Positive Prompt": "a_prompt",
        "Default Negative Prompt": "n_prompt",
        "Auto-Encoder Data Type": "ae_dtype",
        "Diffusion Data Type": "diff_dtype",
        "Number Of Steps": "edm_steps",
        "Face Options BG restoration": "apply_bg",
        "Face Options Face restoration": "apply_face",
        "Auto Caption With LLaVA": "apply_llava",
        "Upscale With SUPIR": "apply_supir",
        "Used Base Model": "ckpt_select",
        "Checkpoint Type": "ckpt_type",
        "max_megapixels": "max_megapixels",
        "max_resolution": "max_resolution"
    }
    return key_mapping.get(key, key)