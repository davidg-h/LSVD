# https://huggingface.co/tasks/text-to-image
# https://huggingface.co/black-forest-labs/FLUX.1-dev
# https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation
# https://huggingface.co/blog/if
# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img
# https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/text2img

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("reference.png")