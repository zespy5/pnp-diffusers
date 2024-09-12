import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from pnp_utils import *

# suppress partial model loading warning
logging.set_verbosity_error()


class PNP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        # load image
        self.image, self.eps = self.get_data()

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # load image
        image = Image.open(self.config["image_path"]).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        # get noise
        latents_path = os.path.join(self.config["latents_path"],  f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.config["latents_path"])
        
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
            self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
            self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
            register_attention_control_efficient(self, self.qk_injection_timesteps)
            register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run(self):
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        decoded_latent = self.sample_loop(self.eps)
        edited_img = T.ToPILImage()(decoded_latent[0])

        return edited_img
        
    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(f'{self.config["output_path"]}')
                
        return decoded_latent

class PNP_VSP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        # load image
        self.image, self.eps = self.get_data()

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        self.pnp_guidance_embeds = self.get_text_embeds("", "")
        


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # load image
        image = Image.open(self.config["image_path"]).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        # get noise
        latents_path = os.path.join(self.config["latents_path"],  f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.config["latents_path"])
        

        style_source_latents = load_source_latents_t(t, self.config["style_latents_path"])
        latent_model_input = torch.cat([source_latents, style_source_latents] + ([x] * 2))


        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance

        _, _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(4)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t, v_injection_t_start=None, v_injection_t_end=None):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.v_injection_timesteps = self.scheduler.timesteps[v_injection_t_start: v_injection_t_end] if v_injection_t_start < v_injection_t_end else None
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_content_control_efficient(self, self.qk_injection_timesteps)
        register_attention_style_control_efficient(self, self.v_injection_timesteps)
        register_conv_content_control_efficient(self, self.conv_injection_timesteps)

    def run(self):
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])

        style_attn_t_start = int(self.config["n_timesteps"] * self.config["style_attn_t_start"])
        style_attn_t_end = int(self.config["n_timesteps"] * self.config["style_attn_t_end"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t,
                    v_injection_t_start=style_attn_t_start, v_injection_t_end=style_attn_t_end)
        
        decoded_latent = self.sample_loop(self.eps)
        edited_img = T.ToPILImage()(decoded_latent[0])

        return edited_img
        
    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(f'{self.config["output_path"]}')
                
        return decoded_latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_path', type=str, default='results/0.png')
    parser.add_argument('--image_path', type=str, default='data/0.jpg')
    parser.add_argument('--latents_path', type=str, default='latents_forward/0')
    parser.add_argument('--sd_version', type=str, default='2.1')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--n_timesteps', type=int, default=50)
    parser.add_argument('--prompt', type=str, required=True, default='')
    parser.add_argument('--negative_prompt', type=str, default='ugly, blurry, black, low res, unrealistic')
    parser.add_argument('--pnp_attn_t', type=float, default=0.5)
    parser.add_argument('--pnp_f_t', type=float, default=0.5)
    parser.add_argument('--use_style_img', type=bool, default=False)
    parser.add_argument('--style_latents_path', type=str, default='latents_forward/night')
    parser.add_argument('--style_attn_t_start', type=float, default=0.8)
    parser.add_argument('--style_attn_t_end', type=float, default=0.9)
    opt = parser.parse_args()
    
    config = vars(opt)
    seed_everything(config["seed"])
    print(config)
    pnp = PNP_VSP(config)
    pnp.run()