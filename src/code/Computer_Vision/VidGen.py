# script for text-to-video
# https://huggingface.co/docs/diffusers/en/using-diffusers/text-img2vid
# https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video
# https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/text_to_video_zero
# https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video_zero

import os
import torch
from huggingface_hub import login
from diffusers.utils import  export_to_video, load_image
from diffusers import I2VGenXLPipeline, CogVideoXImageToVideoPipeline, StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

class VideoGenerator:
    def __init__(self,
                 reference_path, 
                 prompt, 
                 negative_prompt="bad quality, worse quality, low resolution",
                 fps = 7,
                 num_frames = 49,
                 decode_chunk_size = 2,
                 login_token = ""
                 ):
        
        self.reference_path = reference_path
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.fps = fps
        self.num_frames = num_frames
        self.decode_chunk_size = decode_chunk_size
        
        if login_token != "":
            login(login_token)
            
        self.pipe = None
    
    def __cleanup(self):
        self.pipe.to("cpu")
        del self.pipe
        torch.cuda.empty_cache()
        
    def __optimize(self):
        if self.pipe != None:
            try:
                self.pipe.enable_model_cpu_offload()
                self.pipe.unet.enable_forward_chunking()
            except Exception as e:
                print("Warning: " + e)
        else:
            print("No pipeline initialized yet. No optimization possible")
        
    def I2VGen(self):
        image_url = (
            os.path.join(self.reference_path,"reference.png")
            #"https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png" # hf example
        )
        image = load_image(image_url).convert("RGB")
        
        self.pipe = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        )

        # Optimize
        self.__optimize()

        generator = torch.manual_seed(8888)

        frames = self.pipe(
            prompt=self.prompt,
            num_frames=self.num_frames,
            decode_chunk_size=self.decode_chunk_size,
            negative_prompt=self.negative_prompt,
            image=image,
            num_inference_steps=50,
            guidance_scale=9.0,
            generator=generator,
        ).frames[0]
        export_to_video(frames, os.path.join(self.reference_path,"i2v.mp4"), fps=self.fps)
        
        self.__cleanup()
        
        print(f"Video generated successfully with I2VGen-Pipeline in: {self.reference_path}")
        return True
    
    def CogVideoX(self):
        image = load_image(image= os.path.join(self.reference_path,"reference.png"))
        
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )

        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        # Optimize
        self.__optimize()

        video = self.pipe(
            prompt=self.prompt,
            #decode_chunk_size=self.decode_chunk_size,
            num_frames=self.num_frames,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, os.path.join(self.reference_path, "cogVidX.mp4"), fps=self.fps)
        
        self.__cleanup()
        
        print(f"Video generated successfully with CogVideoX-Pipeline in: {self.reference_path}")
        return True
    
    def StableVideoDiffusion(self):
        image = load_image(
            os.path.join(self.reference_path,"reference.png")
            #"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
            ) # hf example
        image = image.resize((1024, 576))
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )

        # Optimize
        self.__optimize()


        generator = torch.manual_seed(42)
        frames = self.pipe(image, 
                      num_frames=self.num_frames, 
                      decode_chunk_size=8, 
                      generator=generator
                      ).frames[0]
        export_to_video(frames,os.path.join(self.reference_path, "svd.mp4"), fps=self.fps)
        
        self.__cleanup()
        
        print(f"Video generated successfully with StableVideoDiffusion-Pipeline in: {self.reference_path}")
        return True
        
    def AnimateDiff(self):
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

        self.pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(
            "emilianJR/epiCRealism",
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        self.pipe.scheduler = scheduler
        self.pipe.enable_vae_slicing()

        # Optimize
        self.__optimize()

        output = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_frames=32,
            decode_chunk_size=self.decode_chunk_size,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=torch.Generator("cpu").manual_seed(49),
        )
        frames = output.frames[0]
        export_to_video(frames, os.path.join(self.reference_path, "animateDiff.mp4"), fps=self.fps)

        self.__cleanup()
        
        print(f"Video generated successfully with AnimateDiff-Pipeline in: {self.reference_path}")
        return True
        
    def ModelscopeT2V(self):

        self.pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        self.pipe.enable_vae_slicing()

        # Optimize
        self.__optimize()

        video_frames = self.pipe(
            self.prompt,
            num_frames=self.num_frames,
            ).frames[0]
        export_to_video(video_frames, os.path.join(self.reference_path,"modelscopet2v.mp4"), fps=self.fps)
        
        self.__cleanup()
        
        print(f"Video generated successfully with ModelscopeT2V-Pipeline in: {self.reference_path}")
        return True