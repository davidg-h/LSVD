import os
import subprocess

#
# Combines the video and audio output into one complete video
#

def install_ffmpeg():
    try:
        # Update the package list
        print("Updating package list...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        
        # Install FFmpeg
        print("Installing FFmpeg...")
        subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        
        print("FFmpeg installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing FFmpeg: {e}")

def fuse_video_audio(video_file, audio_file, output_file):
    try:
        # Use FFmpeg to merge video and audio
        print(f"Merging {video_file} and {audio_file} into {output_file}...")
        subprocess.run(["ffmpeg", "-i", video_file, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_file], check=True)
        print(f"Successfully merged into {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while merging video and audio: {e}")
