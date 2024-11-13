import os
from Text_to_Audio import Text2Audio
from Computer_Vision import Text2Img, VidGen
from Video_Audio import VidAudFusion

prompt = "A happy woman walking down an alley with ice cream in her hand"
path = os.getcwd()

_ , new_prompt = Text2Audio.text2audio(prompt, path)

Text2Img.text2img(new_prompt, path)

generator = VidGen.VideoGenerator(path, new_prompt)

generator.AnimateDiff()
# generator.CogVideoX()
# generator.I2VGen()
# generator.StableVideoDiffusion()
# generator.ModelscopeT2V()

audio = os.path.join(path, "musicgen_out.wav")
video = os.path.join(path, "video file")
output = os.path.join(path, "complete_out.mp4")

VidAudFusion.install_ffmpeg()
VidAudFusion.fuse_video_audio(video, audio, output)