# # Running Diffusers diffusion-dpo training on Modal

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from modal import (
    Image,
    Secret,
    Stub,
    Volume,
    asgi_app,
    enter,
    gpu,
    method,
)

GIT_SHA = "53c508af53870b16910e658c8440dce80e1e5565"
# "5784797c5502e5308a4eaaae705830ff1350eb18" # previous workin sha

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.21.0",
        "datasets~=2.17.1",
        "diffusers",
        "ftfy~=6.1.1",
        "gradio~=3.50.2",
        "smart_open~=6.4.0",
        "transformers==4.37.0",
        # "safetensors==0.2.8",
        "torch~=2.2.0",
        "bitsandbytes",
        "torchvision",
        "triton~=2.2.0",
        "xformers==0.0.24",
        "PyGithub>=1.59",
        "GitPython",
        "Jinja2",
        "peft==0.8.0",
        "wandb",
        "tensorboard",
    )
    .apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's current working directory, /root.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/nikdnaik/diffusers.git",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
    )
)

# ## Set up `Volume`s for training data and model output
#
# Modal can't access your local filesystem, so you should set up a `Volume` to eventually save the model once training is finished.

web_app = FastAPI()
stub = Stub(name="dpo-diffusers-app")

# MODEL_DIR = Path("/sd-1.5-dpo-model")
# training_data_volume = Volume.from_name(
#     "sd-1.5-dpo-diffusers-training-data-volume", create_if_missing=True
# )
# model_volume = Volume.from_name(
#     "sd-1.5-dpo-diffusers-model-volume", create_if_missing=True
# )

# VOLUME_CONFIG = {
#     "/training_data": training_data_volume,
#     "/sd-1.5-dpo-model": model_volume,
# }

MODEL_DIR = Path("/model")
training_data_volume = Volume.from_name(
    "diffusers-training-data-volume", create_if_missing=True
)
model_volume = Volume.from_name(
    "text_diffuser_MARIOEval_2000_steps_lr_1e-6_beta_5000", create_if_missing=True
)

VOLUME_CONFIG = {
    "/training_data": training_data_volume,
    "/model": model_volume,
}



# ## Set up config


@dataclass
class TrainConfig:
    """Configuration for the finetuning training."""

    # identifier for pretrained model on Hugging Face
    # model_name: str = "runwayml/stable-diffusion-v1-5"
    model_name: str ="runwayml/stable-diffusion-v1-5"

    resume_from_checkpoint: str = "latest"
    # HuggingFace Hub dataset
    dataset_name = "nikdnaik/MARIOEval" #"kashif/pickascore"
    dataset_split_name = "train" #"validation" for #kashif
    logging_dir = "log_MARIOEval_2000_steps_lr_1e-6_beta_5000"

    mixed_precision: str = 'fp16'  # set the precision of floats during training, fp16 or less needs to be mixed with fp32 under the hood
    resolution: int = 512  # how big should the generated images be?
    max_train_steps: int = 2000  # number of times to apply a gradient update during training -- increase for better results on the heroicon dataset
    checkpointing_steps: int = 100
    train_batch_size: int = 64 #8
    gradient_accumulation_steps: int = 1  

    learning_rate: float = 1e-06  # scaling factor on gradient updates, make this proportional to the batch size * accumulation steps
    lr_scheduler: str = (
        "constant"  # dynamic schedule for changes to the base learning_rate
    )

    lr_warmup_steps: int = 0  # for non-constant lr schedules, how many steps to spend increasing the learning_rate from a small initial value
    max_grad_norm: int = (
        1  # value above which to clip gradients, stabilizes training
    )
    
    validation_steps: int = 100

    seed: str = '0'

    beta_dpo: int = 5000


@dataclass
class AppConfig:
    """Configuration information for inference."""

    num_inference_steps: int = 20
    guidance_scale: float = 7.5


# The `@stub.function` decorator takes several arguments, including:
# - `image` - the Docker image that you want to use for training. In this case, we are using the `image` object that we defined above.
# - `gpu` - the type of GPU you want to use for training. This argument is optional, but if you don't specify a GPU, Modal will use a CPU.
# - `mounts` - the local directories that you want to mount to the Modal container. In this case, we are mounting the local directory where the training images reside.
# - `volumes` - the Modal volumes that you want to mount to the Modal container. In this case, we are mounting the `Volume` that we defined above.
# - `timeout` argument - an integer representing the number of seconds that the training job should run for. This argument is optional, but if you don't specify a timeout, Modal will use a default timeout of 300 seconds, or 5 minutes. The timeout argument has an upper limit of 24 hours.
# - `secrets` - the Modal secrets that you want to mount to the Modal container. In this case, we are mounting the HuggingFace API token secret.
@stub.function(
    image=image,
    gpu=gpu.A100(size="80GB"),  # finetuning is VRAM hungry, so this should be an A100 or H100
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,  # enables saving of larger files on Modal Volumes
    timeout=60 * 60 * 6,  # two hours, for longer training jobs
    secrets=[Secret.from_name("nn-huggingface-secret"), Secret.from_name("nn-github-secret")],
)
# ## Define the training function
# Now, finally, we define the training function itself.
# This training function does a bunch of preparatory things, but the core of the work is in the training script.
# Depending on which Diffusers script you are using, you will want to modify the script name, and the arguments that are passed to it.
def train():
    import huggingface_hub
    from accelerate import notebook_launcher
    from accelerate.utils import write_basic_config

    # change this line to import the training script you want to use
    from examples.research_projects.diffusion_dpo.train_diffusion_dpo import main
    from transformers import CLIPTokenizer

    # set up TrainConfig
    config = TrainConfig()

    # set up runner-local image and shared model weight directories
    os.makedirs(MODEL_DIR, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # authenticate to hugging face so we can download the model weights
    hf_key = os.environ["HF_TOKEN"]
    huggingface_hub.login(hf_key)

    # check whether we can access the model repo
    try:
        CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    except OSError as e:  # handle error raised when license is not accepted
        license_error_msg = f"Unable to load tokenizer. Access to this model requires acceptance of the license on Hugging Face here: https://huggingface.co/{config.model_name}."
        raise Exception(license_error_msg) from e

    def launch_training():
        sys.argv = [
            "examples/research_projects/diffusion_dpo/train_diffusion_dpo.py", 
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--output_dir={MODEL_DIR}", 
            f"--mixed_precision={config.mixed_precision}",
            f"--dataset_name={config.dataset_name}",
            f"--dataset_split_name={config.dataset_split_name}",
            f"--resolution=512",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--gradient_checkpointing",
            f"--use_8bit_adam",
            f"--rank=8",
            f"--learning_rate={config.learning_rate}", #1e-5",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps=0",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--run_validation",
            f"--validation_steps={config.validation_steps}",
            f"--seed={config.seed}",
            f"--logging_dir={config.logging_dir}",
            f"--beta_dpo={config.beta_dpo}",
            # f"--resume_from_checkpoint={config.resume_from_checkpoint}",
            # f"--report_to=\"wandb\"",
            # f"--push_to_hub",
        ]
        main()



    # run training -- see huggingface accelerate docs for details
    print("launching fine-tuning training script")

    notebook_launcher(launch_training, num_processes=1)

    print('training finished. Model dir Contents:')
    print(os.listdir(os.path.join(MODEL_DIR)))

    print(os.listdir(os.path.join(MODEL_DIR, f"checkpoint-{config.max_train_steps}")))

    # The trained model artefacts have been output to the volume mounted at `MODEL_DIR`.
    # To persist these artefacts for use in future inference function calls, we 'commit' the changes
    # to the volume.
    model_volume.commit()


@stub.local_entrypoint()
def run():
    train.remote()


# ## Run training function
#
# To run this training function:
#
# ```bash
# modal run train_and_serve_diffusers_script.py
# ```
#
# ## Set up inference function
#
# Depending on which Diffusers training script you are using, you may need to use an alternative pipeline to `StableDiffusionPipeline`. The READMEs of the example training scripts will generally provide instructions for which inference pipeline to use. For example, if you are fine-tuning Kandinsky, it tells you to use [`AutoPipelineForText2Image`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky#diffusers.AutoPipelineForText2Image) instead of `StableDiffusionPipeline`.


@stub.cls(
    image=image,
    gpu="A10G",  # inference requires less VRAM than training, so we can use a cheaper GPU
    volumes=VOLUME_CONFIG,  # mount the location where your model weights were saved to
)
class Model:
    @enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        # DDIMScheduler, StableDiffusionPipeline

        # Reload the modal.Volume to ensure the latest state is accessible.
        model_volume.reload()
        config = TrainConfig()
        # set up a hugging face inference pipeline using our model
        load_path = os.path.join(MODEL_DIR, f"checkpoint-{config.max_train_steps}")
        # ddim = DDIMScheduler.from_pretrained(load_path, subfolder="scheduler")
        # # potentially use different pipeline
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     load_path,
        #     scheduler=ddim,
        #     torch_dtype=torch.float16,
        #     safety_checker=None,
        # ).to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()

        weight_dtype = torch.float32
        if config.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif config.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        pipe = DiffusionPipeline.from_pretrained(
        config.model_name,
        torch_dtype=weight_dtype,
        )
        print(os.listdir(MODEL_DIR))

        pipe.load_lora_weights(MODEL_DIR, weight_name="pytorch_lora_weights.safetensors")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

    @method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image


# ## Set up Gradio app
#
# Finally, we set up a Gradio app that will allow you to interact with your model. This will be mounted to the Modal container, and will be accessible at the URL of your Modal deployment. You can refer to the [Gradio docs](https://www.gradio.app/docs/interface) for more information on how to customize the app.


@stub.function(
    image=image,
    concurrency_limit=3,
)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    # Call to the GPU inference function on Modal.
    def go(text):
        return Model().inference.remote(text, config)

    # set up AppConfig
    config = AppConfig()


    example_prompts = [
        f"a movie ticket",
        f"Barack Obama",
        f"a castle",
        f"a German Shepherd",
    ]

    description = f"""Diffusion DPO demo."""

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title="Generate custom heroicons",
        examples=example_prompts,
        description=description,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# ## Run Gradio app
#
# Finally, we run the Gradio app. This will launch the Gradio app on Modal.
#
# ```bash
# modal serve train_diffusion_dpo_modal.py
# ```
