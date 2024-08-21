import argparse
from pnp import PNP_VSP
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--use_style_img', type=bool, default=False)
    # data
    parser.add_argument('--latents_path', type=str, required=True, default='latents_forward')
    parser.add_argument('--style_latents_path', type=str, default='')
    # diffusion
    parser.add_argument('--sd_version', type=str, default='2.1')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--n_timesteps', type=int, default=50)
    parser.add_argument('--negative_prompt', type=str, default='ugly, blurry, low res, unrealistic, paint, black')
    
    #injection threshold
    parser.add_argument('--pnp_attn_t', type=float, default=0.5)
    parser.add_argument('--pnp_f_t', type=float, default=0.5)
    parser.add_argument('--style_attn_t_start', type=float, default=0.8)
    parser.add_argument('--style_attn_t_end', type=float, default=0.9)
    
    config = vars(parser.parse_args())  
    
    #run
    origin_img = Image.open(config['image_path']).resize((512,910))
    pnp_vsp = PNP_VSP(config)
    edited_img = pnp_vsp.run_pnp()
    
    fig = plt.figure(figsize=(20,5))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(origin_img)
    ax1.set_title('origin image')
    ax1.axis("off")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(edited_img)
    ax2.set_title("edited image\n"+config['prompt'])
    ax2.axis("off")
    
    plt.show()
    plt.close()