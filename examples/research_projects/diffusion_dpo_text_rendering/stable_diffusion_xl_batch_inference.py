# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#

# ## Basic setup

from io import BytesIO
from pathlib import Path
import pandas as pd 

from modal import Image, Stub, build, enter, gpu, method

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


image = Image.debian_slim().pip_install(
    "Pillow~=10.1.0",
    "diffusers~=0.24.0",
    "transformers~=4.35.2",  # This is needed for `import torch`
    "accelerate~=0.25.0",  # Allows `device_map="auto"``, which allows computation of optimized device_map
    "safetensors~=0.4.1",  # Enables safetensor format as opposed to using unsafe pickle format
    "pandas",
)

stub = Stub("stable-diffusion-xl-turbo-batch", image=image)

with image.imports():
    import torch
    from diffusers import AutoPipelineForText2Image
    from diffusers.utils import load_image
    from huggingface_hub import snapshot_download
    from PIL import Image

# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@stub.cls(gpu=gpu.A100(), container_idle_timeout=1200)
class Model:
    @build()
    def download_models(self):
        # Ignore files that we don't need to speed up download time.
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]

        snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)

    @enter()
    def enter(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            device_map="auto",
        )

    @method()
    def inference(self, prompt):
        num_inference_steps = 4
        # "When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1"
        # See: https://huggingface.co/stabilityai/sdxl-turbo

        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes



# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --prompt "An astronaut riding a green horse"`


@stub.local_entrypoint()
def main():

    datasets = ['MARIOEval/OpenLibraryEval500/OpenLibraryEval500.txt', 'MARIOEval/TMDBEval500/TMDBEval500.txt', 'MARIOEval/DrawBenchText/DrawBenchText.txt']
    output_dirs = ['MARIOEval/OpenLibraryEval500/images_sdxl', 'MARIOEval/TMDBEval500/images_sdxl', 'MARIOEval/DrawBenchText/images_sdxl']
    import pandas as pd
    for i, dataset in enumerate(datasets): 
        prompts = pd.read_csv(dataset, header=None)
        print(dataset, ' ', len(prompts))
        dir = Path(output_dirs[i])

        if not dir.exists():
            dir.mkdir(exist_ok=True, parents=True)

        for j, prompt in enumerate(prompts[0]):
            print(j, ' ', prompt)
            image_bytes = Model().inference.remote(prompt)
            print(image_bytes)
            output_path = dir / f"{j:04}.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.
