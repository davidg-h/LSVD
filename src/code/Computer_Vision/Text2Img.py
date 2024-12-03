# https://huggingface.co/tasks/text-to-image
# https://huggingface.co/black-forest-labs/FLUX.1-dev
# https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation
# https://huggingface.co/blog/if
# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/text2img
# https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/text2img

import os
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def text2img(prompt, output_path):
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    image = pipe(prompt).images[0]

    image.save(os.path.join(output_path, "reference.png"))

    # cleanup
    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()
    
    print(f"Image generated successfully in: {output_path}")
    return True