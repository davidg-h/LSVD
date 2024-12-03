import os
import sys
from Text_to_Audio import Text2Audio
from Computer_Vision import Text2Img, VidGen
from Video_Audio import VidAudFusion


# command line prompt entry
prompt = ''
if len(sys.argv) > 1:
    prompt = sys.argv[1]

# default prompt
if prompt == '':
    prompt = "A happy woman walking down an alley with ice cream in her hand"
path = os.getcwd()

print('generating audio...')
_ , new_prompt = Text2Audio.text2audio(prompt, path)


print('generating reference image...')
Text2Img.text2img(new_prompt, path)

print('generating video...')
generator = VidGen.VideoGenerator(path, new_prompt)

# generator.AnimateDiff()
# generator.CogVideoX()
# generator.I2VGen()
generator.StableVideoDiffusion()
# generator.ModelscopeT2V()
audio = os.path.join(path, "musicgen_out.wav")
video = os.path.join(path, "svd.mp4") # animateDiff.mp4
output = os.path.join(path, "complete_out.mp4")

print('merging audio and video...')
VidAudFusion.install_ffmpeg()
VidAudFusion.fuse_video_audio(video, audio, output)

print(f'final advertisement found at: {output}')