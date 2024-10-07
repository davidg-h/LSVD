# script for text-to-video
# https://huggingface.co/docs/diffusers/en/using-diffusers/text-img2vid
# https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video
# https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/text_to_video_zero
# https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video_zero

import torch
from huggingface_hub import login
from diffusers import I2VGenXLPipeline
from diffusers.utils import  export_to_video, load_image

login("your_token")

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8888)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator
).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
