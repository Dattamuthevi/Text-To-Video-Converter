from flask import Flask, render_template, request, send_file
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline, DPMSolverMultistepScheduler, export_to_video
from transformers import pipeline
import torch
from google.colab import drive
from IPython.display import HTML
from pathlib import Path
from flask_ngrok import run_with_ngrok
import os
from google.colab import files

app = Flask(__name__)
run_with_ngrok(app)  # Starts ngrok when the app is run

# Set up Diffusers model
CFG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "image_gen_steps": 35,
    "image_gen_model_id": "CompVis/stable-diffusion-v1-4",
    "image_gen_guidance_scale": 9,
    "prompt_gen_model_id": "gpt2",
    "prompt_dataset_size": 6,
    "prompt_max_length": 12
}

scheduler = EulerDiscreteScheduler.from_pretrained(
    CFG["image_gen_model_id"], subfolder="scheduler")

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG["image_gen_model_id"],
    torch_dtype=torch.float32,
    scheduler=scheduler,
    revision="fp16",
    use_auth_token='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',
    guidance_scale=CFG["image_gen_guidance_scale"]
)

image_gen_model = image_gen_model.to(CFG["device"])

# Routes for image generation
@app.route('/generate_image', methods=['POST'])
def generate_image():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = generate_image_from_prompt(prompt)
        return send_file(image, mimetype='image/jpeg')

def generate_image_from_prompt(prompt):
    image = image_gen_model(
        prompt,
        num_inference_steps=CFG["image_gen_steps"],
        generator=torch.Generator(device=CFG["device"]),
        guidance_scale=CFG["image_gen_guidance_scale"]
    ).images[0]
    image.save("generated_image.jpg")  # Save the generated image
    return "generated_image.jpg"

# Routes for video generation
@app.route('/generate_video', methods=['POST'])
def generate_video():
    if request.method == 'POST':
        prompt = request.form['prompt']
        video_path = generate_video_from_prompt(prompt)
        return send_file(video_path, mimetype='video/mp4')

def generate_video_from_prompt(prompt):
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    video_frames = pipe(prompt, num_inference_steps=25, num_frames=20).frames
    video_path = export_to_video(video_frames)
    return video_path

if __name__ == '__main__':
    app.run()
