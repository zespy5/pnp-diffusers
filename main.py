import argparse
import torch
from pnp import PNP_VSP, PNP
from preprocess import run
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_prompt(images):
  
  text = "a photography of"
  inputs = processor(images, text, return_tensors="pt").to("cuda", torch.float16)

  out = model.generate(**inputs)
  caption = processor.decode(out[0], skip_special_tokens=True)
  is_night = False
  if 'at night ' in caption:
      caption = caption.split('at night ')[0]
      is_night = True
      return caption, is_night
  
  return caption, is_night

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        default='data/0.jpg')
    parser.add_argument('--save_dir', type=str, default='latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=999)
    parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    
    # general
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--use_style_img', type=bool, default=False)
    # data
    parser.add_argument('--latents_path', type=str)
    parser.add_argument('--style_latents_path', type=str, default='')
    # diffusion
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--n_timesteps', type=int, default=50)
    parser.add_argument('--negative_prompt', type=str, default='ugly, blurry, low res, unrealistic, paint')
    
    #injection threshold
    parser.add_argument('--pnp_attn_t', type=float, default=0.5)
    parser.add_argument('--pnp_f_t', type=float, default=0.5)
    parser.add_argument('--style_attn_t_start', type=float, default=0.8)
    parser.add_argument('--style_attn_t_end', type=float, default=0.9)
    
    opt = parser.parse_args()
    config = vars(parser.parse_args())
    config['image_path'] = opt.data_dir
    pathlib_img = Path(opt.data_dir)
    config['latents_path'] = 'latents_forward/'+pathlib_img.stem
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

    origin_img = Image.open(config['image_path'])
    prompt, is_night = generate_prompt(origin_img)
    Path('results').mkdir(exist_ok=True)
    if is_night:
        
        original_output_path = config['output_path']
        original_guidance_scale = config['guidance_scale']
        
        data_file = Path(opt.data_dir)
        latent_file = Path(opt.save_dir+'_forward')/data_file.stem/'noisy_latents_1.pt'
        if latent_file.exists():
            print('There exist night latent files')
        else:
            run(opt)
        edited_prompt = prompt+'at daytime on a sunny day'
        print(edited_prompt)
        config['prompt'] = edited_prompt
        
        n2d_dir = Path('n2d_results')
        n2d_dir.mkdir(exist_ok=True)
        intermediate_img_save = n2d_dir/(data_file.stem+'2day.png')
        refresh = False
        if not intermediate_img_save.exists():
            config['output_path'] = intermediate_img_save.as_posix()
            config['guidance_scale'] = 30.0
            pnp = PNP_VSP(config) if config['use_style_img'] else PNP(config)
            pnp.run()
            refresh = True
            
        
        #d2other
        n2d_data_file = config['output_path']
        latent_file = Path(opt.save_dir+'_forward')/intermediate_img_save.stem/'noisy_latents_1.pt'
        
        if latent_file.exists() and not refresh:
            print('There exist day latent files')
        else:
            opt.data_dir = n2d_data_file
            run(opt)
        config['output_path'] = original_output_path
        config['guidance_scale'] = original_guidance_scale
        config['latents_path'] = 'latents_forward/'+intermediate_img_save.stem
        edited_prompt = prompt+config['condition']
        print(edited_prompt)
        config['prompt'] = edited_prompt
        pnp = PNP_VSP(config) if config['use_style_img'] else PNP(config)
        pnp.run()
        
    else:
        data_file = Path(opt.data_dir)
        latent_file = Path(opt.save_dir+'_forward')/data_file.stem/'noisy_latents_1.pt'
        if latent_file.exists():
            print('There exist latent files')
        else:
            run(opt)
        edited_prompt = prompt+config['condition']
        print(edited_prompt)
        config['prompt'] = edited_prompt
        pnp = PNP_VSP(config) if config['use_style_img'] else PNP(config)
        pnp.run()
        
    
        
    
