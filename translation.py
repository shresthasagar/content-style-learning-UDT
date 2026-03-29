import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
import pickle
from tqdm import tqdm, trange
import random
import sys

def logprint(*args, verbose=True):
    if verbose:
        print(*args)

class Translation:
    def __init__(self, generator_path, c_dim, i_dim, num_steps=1000, psi=0.7, device='cuda', num_styles_per_content=1, initial_learning_rate=0.1, use_noise=True, random_start=False):
        self.device = device
        self.c_dim = c_dim
        self.i_dim = i_dim
        self.num_steps = num_steps
        self.psi = psi
        self.num_styles_per_content = num_styles_per_content
        self._load_networks(generator_path)
        self._compute_stats()
        self.initial_learning_rate = initial_learning_rate
        self.use_noise = use_noise
        self.random_start = random_start
        self.fixed_style = torch.randn(1, self.G.z_dim).cuda()

    def _load_networks(self, generator_path):
        print('Loading networks from "%s"...' % generator_path)
        device = torch.device('cuda')
        with dnnlib.util.open_url(generator_path) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)

    def _compute_stats(self):
        w_avg_samples = 10000
        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)

        c = torch.zeros([z_samples.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
        cum_num_samples_per_class = [z_samples.shape[0]//self.c_dim] * (self.c_dim-1) + [z_samples.shape[0] - (z_samples.shape[0]//self.c_dim)*(self.c_dim-1)]
        idx_class = [0] + list(np.cumsum(cum_num_samples_per_class))
        for i in range(self.c_dim):
            c[idx_class[i]:idx_class[i+1], i] = 1
        
        latent_samples = self.G.mapping(torch.from_numpy(z_samples).to(self.device), c).cpu().numpy().astype(np.float32) # [N, num_ws, C]
        content, style = latent_samples[:, 0], latent_samples[:, -1]
        styles = [style[idx_class[i]:idx_class[i+1]] for i in range(self.c_dim)]

        self.latent_avg = np.mean(latent_samples, axis=0, keepdims=True)      # [1, num_ws, C]
        self.latent_std = (np.sum((latent_samples - self.latent_avg) ** 2) / w_avg_samples) ** 0.5
        self.content_avg = np.mean(content, axis=0, keepdims=True)      # [1, C]
        self.styles_avg = [np.mean(s, axis=0, keepdims=True) for s in styles]      # [1, C]
        self.content_std = (np.sum((content - self.content_avg) ** 2) / w_avg_samples) ** 0.5
        self.styles_std = [(np.sum((s - self.styles_avg[i]) ** 2) / cum_num_samples_per_class[i] )** 0.5 for i, s in enumerate(styles)] # [1, C]
        
        print('content_std', self.content_std)
        print('styles_std', self.styles_std)
        print('latent_std', self.latent_std)
        print('cum num_samples per class', cum_num_samples_per_class)

    def refresh_style(self):
        self.fixed_style = torch.randn(1, self.G.z_dim).cuda()

    def get_ws_from_c_s(self, c, s):
        return np.concatenate([np.repeat(np.expand_dims(c, axis=1), self.G.mapping.num_c_res, axis=1),
                             np.repeat(np.expand_dims(s, axis=1), self.G.num_ws-self.G.mapping.num_c_res, axis=1)], axis=1)

    def get_ws_from_c_s_torch(self, c, s):
        if len(c.shape) > 2:
            return torch.cat([c,
                          s.unsqueeze(1).repeat(1, self.G.num_ws-self.G.mapping.num_c_res, 1)], axis=1)

        return torch.cat([c.unsqueeze(1).repeat(1, self.G.mapping.num_c_res, 1),
                          s.unsqueeze(1).repeat(1, self.G.num_ws-self.G.mapping.num_c_res, 1)], axis=1)
    
    
    def project(self, img, cls=None):
        '''
        Takes in a batch of content images and style images and returns the corresponding latent vectors
        '''
        if cls is not None:
            latent_init = self.get_ws_from_c_s(self.content_avg, self.styles_avg[cls]) # [1, num_ws, C]
        else:
            latent_init = self.latent_avg
        latent_init = np.repeat(latent_init, img.shape[0], axis=0) # [N, num_ws, C]
        if self.random_start:
            c = torch.zeros([latent_init.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
            c[:, cls] = 1
            latent_init = self.G.mapping(torch.randn(img.shape[0], self.G.z_dim).to(self.device), c).cpu().numpy().astype(np.float32) # [N, num_ws, C]

        # Features for target image.
        img = img.to(self.device).to(torch.float32).to(self.device)
        img_features = self.vgg16(img, resize_images=False, return_lpips=True)
        latent = self.optimize_latent(latent_init, img_features, self.latent_std, img)
        
        return latent
        # yield from self.optimize_latent(latent_init, content_img_features, self.latent_std, content_images)


    def postprocess(self, content_hat, target_class, fixed_style=False):
        ''' Combines content with a random style from target_class'''
        
        # randomly sample the style vector
        if not fixed_style:
            random_z = torch.randn(content_hat.shape[0], self.G.z_dim, device=self.device)
        else:
            random_z = self.fixed_style.repeat(content_hat.shape[0], 1)
        target_c = torch.zeros([content_hat.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
        target_c[:, target_class] = 1
        latent = self.G.mapping(random_z, target_c)
        style_hat = latent[:, -1]
        if self.psi is not None:
            style_hat = (1-self.psi) * torch.tensor(self.styles_avg[target_class]).to(self.device).type(torch.float32) + self.psi * style_hat
                
        # Combine style and content latents.
        return self.get_ws_from_c_s_torch(content_hat, style_hat)

    def synthesize(self, latent_hat):
        '''
        Takes in a batch of latent vectors and returns the corresponding images
        '''
        synth_images = self.G.synthesis(latent_hat, noise_mode='const')
        return synth_images

    def optimize_latent(
            self,
            w_avg,
            target_features,
            w_std,
            content_images,
            initial_noise_factor    = 0.05,
            noise_ramp_length       = 0.75,
            lr_rampdown_length      = 0.25,
            lr_rampup_length        = 0.05,
            initial_learning_rate   = 0.1,
            is_style=False
    ):
        initial_learning_rate = self.initial_learning_rate
        
        content = torch.tensor(w_avg[:, 0].copy(), dtype=torch.float32, device=self.device, requires_grad=True) 
        style = torch.tensor(w_avg[:, -1].copy(), dtype=torch.float32, device=self.device, requires_grad=True)
        
        # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True) # pylint: disable=not-callable
        optimizer = torch.optim.Adam([content, style], betas=(0.9, 0.999), lr=initial_learning_rate)
        
        for step in trange(self.num_steps):
            w_opt = self.get_ws_from_c_s_torch(content, style)
            # Learning rate schedule.
            t = step / self.num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            if self.use_noise:
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                w_in = w_opt + w_noise
            else:
                w_in = w_opt
            
            synth_images = self.G.synthesis(w_in, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
            loss = (target_features - synth_features).square().sum() / synth_features.shape[0]
            # loss = torch.norm(synth_images - content_images)/(255*synth_features.shape[0])
            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # logprint(f'step {step+1:>4d}/{self.num_steps}: dist {loss:<4.2f}')
            # y = w_opt.clone()
            # yield y.detach()
        return w_opt.detach()
    

    def get_batch(self, filenames):
        images = []
        for target_fname in filenames:
            target_pil = PIL.Image.open(target_fname).convert('RGB')
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), PIL.Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)
            images.append(target_uint8.transpose([2, 0, 1]))
        return np.stack(images, axis=0)

#----------------------------------------------------------------------------

def postprocess_and_save(trans_img, outdir, counter):
    trans_img = (trans_img + 1.0) * (255 / 2.0)
    trans_img = trans_img.clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    
    for j in range(len(trans_img)):
        imageio.imwrite(f'{outdir}/{counter}_{j}.png', trans_img[j])

#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--content', 'content_dir', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--style', 'style_dir',   help='Target image file to project to', required=False, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=200, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--batchsize',              help='Batch size for generator', type=int, default=32, show_default=True)
@click.option('--psi',                    help='Truncation psi', type=float, default=None, show_default=True)
@click.option('--source_class',             help='Source class for the projection', type=int, default=None, show_default=True)
@click.option('--target_class',             help='Target class for the projection', type=int, default=None, show_default=True)
@click.option('--num_styles_per_content',   help='Number of style vectors per content vector', type=int, default=1, show_default=True)
@click.option('--c_dim',                    type=int, default=3, show_default=True)
@click.option('--from_projection',          help='If true, will run from projection', type=bool, default=False)
@click.option('--projection_pkl',              help='projection pkl', type=str, default='')
@click.option('--save_img_to_pkl',              help='save images to pkl', type=bool, default=False)
@click.option('--img_pkl',                  help='save images to pkl', type=str, default='')
@click.option('--atob',                  help='direction of translation', type=bool, default=True)
@click.option('--max_images',                  help='max images to generate', type=int, default=1000)

def run_translation(
    network_pkl: str,
    content_dir: str,
    style_dir: str,
    num_steps: int,
    seed: int,
    outdir: str,
    batchsize: int,
    psi: float,
    source_class: int,
    target_class: int, 
    num_styles_per_content: int,
    c_dim: int,
    from_projection: bool,
    projection_pkl: str,
    save_img_to_pkl: bool,
    img_pkl: str,
    atob: bool,
    max_images: int
):
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    counter = 0
    num_styles = num_styles_per_content
    translation = Translation(network_pkl, c_dim=c_dim, i_dim=16, num_steps=num_steps, psi=psi, num_styles_per_content=num_styles_per_content, initial_learning_rate=0.01, use_noise=True)

    content_files = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir)])
    if style_dir is not None:
        style_files = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir)])
        max_len = min(len(content_files), len(style_files))
        content_files = content_files[:max_len]
        style_files = style_files[:max_len]
        if not save_img_to_pkl:
            random.shuffle(style_files)
    else:
        style_files = None
    if from_projection:
        print('getting projections')
        with open(projection_pkl, 'rb') as f:
            projections = pickle.load(f)
    
    if save_img_to_pkl:
        if os.path.exists(img_pkl):
            with open(img_pkl, 'rb') as f:
                save_dict = pickle.load(f)
            fill_dummy = False
        else:
            fill_dummy = True
            save_dict = {'real_A': [], 'real_B': [], 'fake_A': [], 'fake_B': [], 'rec_A': [], 'rec_B': [], 'ref_A': [], 'ref_B':[]}

        if atob:
            save_dict['real_A'] = []
            save_dict['fake_B'] = []
            save_dict['rec_A'] = []
            save_dict['ref_A'] = []
        else:
            save_dict['real_B'] = []
            save_dict['fake_A'] = []
            save_dict['rec_B'] = []
            save_dict['ref_B'] = []
    
    # sys.exit()
    if not save_img_to_pkl:
        random.shuffle(content_files)
    for batch_start in trange(0, min(len(content_files), max_images), batchsize):
        if not from_projection:
            content_images = torch.tensor(translation.get_batch(content_files[batch_start:batch_start + batchsize]), device=translation.device)
            latent_content = translation.project(
                            img=content_images,
                            cls=source_class)
            content = latent_content[:,0]
            if style_files is not None:
                style_images = torch.tensor(translation.get_batch(style_files[batch_start:batch_start + batchsize]),device=translation.device) if style_files is not None else None
                latent_style = translation.project(
                                img=style_images,
                                cls=target_class)
                style = latent_style[:,-1]
        else:
            latent_content = torch.stack([torch.tensor(projections[fname]) for fname in content_files[batch_start:batch_start + batchsize]]).to(translation.device)
            content = latent_content[:,0]
            if style_files is not None:
                latent_style = torch.stack([torch.tensor(projections[fname]) for fname in style_files[batch_start:batch_start + batchsize]]).to(translation.device)
                style = latent_style[:,-1]
        if len(content) < num_styles:
            break

        for i, ct in enumerate(content):
            if not save_img_to_pkl:
                if style_files is not None:
                    st = random.sample(list(style), num_styles)
                    st = torch.stack(st, dim=0)
                ct = ct.unsqueeze(0).repeat(num_styles, 1)
            else:
                ct = ct.unsqueeze(0)
                if style_files is not None:
                    st = style[i].unsqueeze(0)

            if style_files is not None:
                latent = translation.get_ws_from_c_s_torch(ct, st)
            else:
                latent = translation.postprocess(ct, target_class)

            trans_img = translation.G.synthesis(latent, noise_mode='const')

            if not save_img_to_pkl:
                postprocess_and_save(trans_img, outdir, counter)
            else:
                trans_img = (trans_img + 1)/2.0
                trans_img = trans_img.clamp(0, 1).to(torch.float32).cpu().numpy()
                
                if atob:
                    save_dict['real_A'].append((content_images[i:i+1]/255.0).cpu())
                    save_dict['fake_B'].append(trans_img[0:1])
                    if style_files is not None:
                        save_dict['ref_A'].append((style_images[i:i+1]/255.0).cpu())
                else:
                    save_dict['real_B'].append((content_images[i:i+1]/255.0).cpu())
                    save_dict['fake_A'].append(trans_img[0:1])
                    if style_files is not None:
                        save_dict['ref_B'].append((style_images[i:i+1]/255.0).cpu())
            
            counter += 1

    if save_img_to_pkl:
        if atob:
            save_dict['real_A'] = np.concatenate(save_dict['real_A'], axis=0)
            save_dict['fake_B'] = np.concatenate(save_dict['fake_B'], axis=0)
            if len(save_dict['ref_A']) > 0:
                save_dict['ref_A'] = np.concatenate(save_dict['ref_A'], axis=0)
        else:
            save_dict['real_B'] = np.concatenate(save_dict['real_B'], axis=0)
            save_dict['fake_A'] = np.concatenate(save_dict['fake_A'], axis=0)
            if len(save_dict['ref_B']) > 0:
                save_dict['ref_B'] = np.concatenate(save_dict['ref_B'], axis=0)
                
        if fill_dummy:
            if atob:
                save_dict['real_B'] = np.zeros_like(save_dict['real_A'])
                save_dict['fake_A'] = np.zeros_like(save_dict['fake_B'])
        
        with open(img_pkl, 'wb') as f:
            pickle.dump(save_dict, f)
            print('Saved images successfully!')


@click.command()
@click.option('--data_path',              help='Datasetpath', required=True)
@click.option('--out_pkl',                help='Where to save the representations', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--data_name',              help='name of the dataset: afhq or celebahq', required=True)
@click.option('--num_steps',              help='Number of optimization steps', type=int, default=200, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--batchsize',              help='Batch size for generator', type=int, default=32, show_default=True)
@click.option('--psi',                    help='Truncation psi', type=float, default=None, show_default=True)
@click.option('--num_styles_per_content',   help='Number of style vectors per content vector', type=int, default=1, show_default=True)
@click.option('--c_dim',                    type=int, default=3, show_default=True)
def save_projections(data_path: str, 
                     out_pkl: str, 
                     network_pkl: str, 
                     data_name: str, 
                     seed: int,
                     num_steps: int, 
                     batchsize: int,
                     psi: float, 
                     num_styles_per_content: int, 
                     c_dim: int):
    
    # extract director path from filename
    outdir = os.path.dirname(out_pkl)
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    domains = {'afhq': ('cat', 'dog', 'wild'),
                'celebahq': ('male', 'female')}
    assert data_name in domains.keys()
    folders = [os.path.join(data_path, f) for f in domains[data_name]]
    
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    translation = Translation(network_pkl, c_dim=c_dim, i_dim=16, num_steps=num_steps, psi=psi, num_styles_per_content=num_styles_per_content, initial_learning_rate=0.01, use_noise=True)
    projections = {}
    for folder in folders:
        print(folder)
        content_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)])
        for batch_start in trange(0, len(content_files), 32):
            content_images = torch.tensor(translation.get_batch(content_files[batch_start:batch_start + 32]), device=translation.device)
            latent_content = translation.project(
                            img=content_images,
                            cls=None)
            for i, fname in enumerate(content_files[batch_start:batch_start + 32]):
                projections[fname] = latent_content[i].cpu().numpy()
    with open(out_pkl, 'wb') as f:
        pickle.dump(projections, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_translation() 

#----------------------------------------------------------------------------
