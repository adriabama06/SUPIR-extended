import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from SUPIR.utils.model_fetch import get_model
from openai_prompt import prompt_prompt


class LLMCaption:
    def __init__(self, device: str, load_8bit: bool = True, load_4bit: bool = False):
        self._model = None
        self._tokenizer = None
        self._device = device
        self._load_8bit = load_8bit
        self._load_4bit = load_4bit

    def load(self):
        model_path = get_model("openbmb/MiniCPM-Llama3-V-2_5")
        quant_config = None
        if self._load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16  # Ensure compute dtype is set for better performance
            )
        elif self._load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        # Load the model and tokenizer here
        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                torch_dtype=torch.float16, quantization_config=quant_config)
        if quant_config is None:
            self._model = self._model.to(device='cuda')
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model.eval()

    def to(self, device):
        if not self._load_4bit and not self._load_8bit:
            self._model.to(device)
        return self

    def caption(self, image, prompt, temp, top_p):
        msgs = [{'role': 'user', 'content': prompt}]
        print(f"Captioning image with prompt: {prompt}, {prompt_prompt}")
        res = self._model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self._tokenizer,
            sampling=True,  # if sampling=False, beam_search will be used by default
            temperature=temp,
            system_prompt=prompt_prompt,
            top_p=top_p
        )
        print(f"Caption: {res}")
        return res


class MoondreamCaption:
    def __init__(self, device: str, load_8bit: bool = True, load_4bit: bool = False):
        self._model = None
        self._tokenizer = None
        self._device = device
        self._load_8bit = load_8bit
        self._load_4bit = load_4bit

    def load(self):
        model_path = get_model("vikhyatk/moondream2")
        self._model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                torch_dtype=torch.float16)
        self._model = self._model.to(device='cuda')
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def to(self, device):
        self._model.to(device)
        return self

    def caption(self, image, prompt, temp, top_p):
        prompt = f"{prompt} - Ensure your answer can be used to generate an image with a Stable Diffusion model."
        print(f"Captioning image with prompt: {prompt}")
        enc_image = self._model.encode_image(image)
        res = self._model.answer_question(enc_image, prompt, self._tokenizer)
        print(f"Caption: {res}")
        return res
