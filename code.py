from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
#pip install diffusers transformers accelerate

import torch

class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "CompVis/stable-diffusion-v1-4"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model_id = "CompVis/stable-diffusion-v1-4"


from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
scheduler =EulerDiscreteScheduler.from_pretrained(image_gen_model_id,subfolder="scheduler")
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float32,
    scheduler=scheduler,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

   #image = image.resize(CFG.image_gen_size)
    return image

generate_image("Kala bhairava fighting with soldiers 4k image", image_gen_model)

#!pip install flask_ngrok
#!pip install pyngrok==4.1.1

#!pip install ngrok

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))


from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    return "Hello World!!"

#@app.route('/display_image/<image_path>')
#def display_image(image_path):
#    return render_template('display_image.html', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)


from google.colab import drive
drive.mount('/content/drive')

#Video Generator


#!pip install accelerate

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
prompt = "loosing the game in the big crowd with 4k quality"
video_frames = pipe(prompt, num_inference_steps=25, num_frames=20).frames

# convent to video
video_path = export_to_video(video_frames)
video_path

from google.colab import files
files.download(video_path)

from IPython.display import HTML

#Embed the video using HTML5 video tags
video_tag = f"<video width='640' height='480' controls><source src='{video_path}' type='video/mp4'></video>"
HTML(video_tag)
