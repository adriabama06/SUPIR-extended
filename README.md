# Docker version of the [fork of SUPIR of FurkanGozukara](https://github.com/FurkanGozukara/SUPIR)

# Installation/Deploy
```bash
# Clone the repository
git clone https://github.com/adriabama06/SUPIR-extended.git
cd SUPIR-extended
# Start the service
docker compose up -d
# Check console
docker compose logs -f
```
The Gradio interface will be available at IP:6688

## Features
- **OpenAI-compatible backend support**: Use OpenAI, Ollama, or TabbyAPI for image captioning (set via environment variables)
- **Option to skip llava model download**: Set `SKIP_LLAVA_DOWNLOAD=on` to use a remote model for captions
- **Docker with GPU support**: Docker Compose is configured for NVIDIA GPUs
- **Automatic model download**: On first run, required models are downloaded unless skipped

## Environment Variables (in `compose.yml`)
- `OPENAI_API_BASE`: Base URL for OpenAI-compatible API (e.g., https://api.openai.com/v1 or Ollama)
- `OPENAI_API_KEY`: API key for OpenAI-compatible backend
- `OPENAI_MODEL`: Model name (e.g., gpt-4o-mini)
- `OPENAI_BACKEND`: Backend type (`ollama`, `tabbyapi`, or `none`) `Note: This is only used for auto-unload the model after the generation of the caption`
- `USE_OPENAI`: `on` to use OpenAI backend, `off` to use local llava
- `SKIP_LLAVA_DOWNLOAD`: `on` to skip local llava download

## Usage Notes
- The entrypoint script checks for model downloads and explains the process if models are missing.
- To use a remote captioning model, set `SKIP_LLAVA_DOWNLOAD=on` and configure the OpenAI backend variables.
- GPU is required (see Docker Compose for device settings).

# Ultra Advanced SUPIR App With So Many Features and 1-Click Install

## Download and use : https://www.patreon.com/posts/99176057

![screencapture-127-0-0-1-7860-2025-04-14-19_22_58](https://github.com/user-attachments/assets/19237667-e5ba-42d6-9716-dfa0061a65c0)
