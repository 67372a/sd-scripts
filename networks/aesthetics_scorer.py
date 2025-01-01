# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py
# and https://github.com/mihirp1998/AlignProp/blob/5e950b3f16ded622df15f4bea2eec93f88962f2b/aesthetic_scorer.py
# and https://github.com/mihirp1998/AlignProp/blob/5e950b3f16ded622df15f4bea2eec93f88962f2b/alignprop_trainer.py

import os

import torch
from torch import nn
import torchvision
import os
import requests
from tqdm import tqdm

from transformers import CLIPModel
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


CLIP_REPO = "openai/clip-vit-large-patch14"
AESTHETIC_MODEL_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MODEL_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"

HPS_V2_MODEL_URL = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
HPS_V2_MODEL_FILENAME = "HPS_v2_compressed.pt"

NORMALIZE_TRANFORM = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

TARGET_SIZE = 224

RESIZE_TRANFORM = torchvision.transforms.Resize(TARGET_SIZE)
CENTER_CROP_TRANFORM = torchvision.transforms.CenterCrop(TARGET_SIZE)

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_REPO)
        self.mlp = MLPDiff()

        # Create the directory if it doesn't exist
        os.makedirs(os.path.expanduser('~/.cache/aesthetics'), exist_ok=True)
        checkpoint_path = f"{os.path.expanduser('~')}/.cache/aesthetics/{AESTHETIC_MODEL_FILENAME}"

        # Download the file if it doesn't exist
        if not os.path.exists(checkpoint_path):
            response = requests.get(AESTHETIC_MODEL_URL, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(checkpoint_path, 'wb') as file, tqdm(
                desc=f"Downloading {AESTHETIC_MODEL_FILENAME}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)

        state_dict = torch.load(checkpoint_path)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

        #pooch.retrieve(
        #    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
        #    "md5:b1047fd767a00134b8fd6529bf19521a",
        #    fname=filename,
        #    path=path,
        #    progressbar=True,
        #)

    def __call__(self, images):
        #device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
    
def aesthetic_loss_fn(aesthetic_target: float = 10.0,
                     grad_scale: float = 1.0,
                     device=None,
                     torch_dtype: torch.dtype = torch.bfloat16):
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    def loss_fn(im_pix_un) -> tuple:
        im_pix = ((im_pix_un / 2.0) + 0.5).clamp(min=0.0, max=1.0) 
        im_pix = CENTER_CROP_TRANFORM(RESIZE_TRANFORM(im_pix))
        im_pix = NORMALIZE_TRANFORM(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def hps_loss_fn(inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/{HPS_V2_MODEL_FILENAME}"

    # Download the file if it doesn't exist
    if not os.path.exists(checkpoint_path):
        response = requests.get(HPS_V2_MODEL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(checkpoint_path, 'wb') as file, tqdm(
            desc=f"Downloading {HPS_V2_MODEL_FILENAME}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.eval()
        
    def loss_fn(im_pix, prompts):    
        im_pix = ((im_pix / 2.0) + 0.5).clamp(min=0.0, max=1.0) 
        x_var = CENTER_CROP_TRANFORM(RESIZE_TRANFORM(im_pix))
        x_var = NORMALIZE_TRANFORM(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return loss, scores
    
    return loss_fn

