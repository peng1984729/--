import os
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "image", "data")   
OUT_DIR = os.path.join(HERE, "image", "result两种风格")
os.makedirs(OUT_DIR, exist_ok = True) 

base_image_path = os.path.join(CACHE_DIR, "content1.jpg")
style1_image_path = os.path.join(CACHE_DIR, "style1.jpg")
style2_image_path = os.path.join(CACHE_DIR, "style2.jpg")
result_prefix = "generation"

total_variation_weight = 1e-7
style_weight = 1e-3
content_weight = 2.5e-8

style1_weight = 0
style2_weight = 1

image_height = 400
iterations = 4000  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_w, _h = Image.open(base_image_path).size
image_width = int(_w * image_height / _h)     
def display(image_path, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(Image.open(image_path))
    plt.title(title)
    plt.axis("off")
    plt.show()

vgg_mean = torch.tensor([103.939, 116.779, 123.68], device=device).view(3, 1, 1)


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC) 
    arr = np.asarray(img, dtype="float32") 
    return torch.from_numpy(arr).permute(2, 0, 1)


def preprocess_image(path):
    x = load_image(path, size=(image_height, image_width)).to(device)
    x = x[[2, 1, 0], :, :]                       
    return (x - vgg_mean).unsqueeze(0)          


def deprocess_image(x):
    x = x.detach().cpu()[0] + vgg_mean.cpu()       
    x = x[[2, 1, 0], :, :]                         
    return x.clamp(0, 255).permute(1, 2, 0).numpy().round().astype("uint8")

vgg = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1).features
vgg = vgg.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

style_layer_idx = [1, 6, 11, 18, 25]            
content_layer_idx = 27
all_idx = sorted(set(style_layer_idx + [content_layer_idx]))
deep = max(all_idx)

def extract_features(x):
    feats = {}
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in all_idx:
            feats[i] = x
        if i >= deep:
            break
    return feats

def gram_matrix(x):
    c = x.shape[1]
    f = x.reshape(c, -1)
    return f @ f.t()

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = image_height * image_width
    return torch.sum((S - C) ** 2) / (4.0 * channels ** 2 * size ** 2)

def S_loss(style1, style2, combination):
    s1 = style1_weight * style_loss(style1, combination)
    s2 = style2_weight * style_loss(style2, combination)
    return s1 + s2

def content_loss(base, combination):
    return torch.sum((base - combination) ** 2)

def total_variation_loss(x):
    h, w = x.shape[2], x.shape[3]
    a = (x[:, :, : h - 1, : w - 1] - x[:, :, 1:, : w - 1]) ** 2
    b = (x[:, :, : h - 1, : w - 1] - x[:, :, : h - 1, 1:]) ** 2
    return torch.sum((a + b) ** 1.25)

def compute_loss(combination_image, base_image, style1_image, style2_image):
    feats = extract_features(torch.cat([base_image, style1_image, style2_image, combination_image], dim = 0))
    loss = torch.zeros((), device = device)
    cf = feats[content_layer_idx]
    loss = loss + content_weight * content_loss(cf[0], cf[3])
    for idx in style_layer_idx:
        lf = feats[idx]
        loss = loss + (style_weight / len(style_layer_idx)) * S_loss(lf[1], lf[2], lf[3])
    loss = loss + total_variation_weight * total_variation_loss(combination_image)
    return loss

display(base_image_path, "Base image")
display(style1_image_path, "Style1 image")
display(style2_image_path, "Style2 image")

base_image = preprocess_image(base_image_path)
style1_image = preprocess_image(style1_image_path)
style2_image = preprocess_image(style2_image_path)
combination_image = base_image.clone().requires_grad_(True)

optimizer = torch.optim.SGD([combination_image], lr = 100.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96 ** (1.0 / 1000))


def save_png(img, fname, tries = 5):
    for k in range(tries):
        try:
            with open(fname, "wb") as f:
                img.save(f, format = "PNG")
            return True
        except OSError as e:
            if k == tries - 1:
                return False
            time.sleep(0.5)


for i in range(iterations + 1):
    optimizer.zero_grad()
    loss = compute_loss(combination_image, base_image, style1_image, style2_image)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 100 == 0:
        print(f"Iteration {i:4d}: loss = {loss.item():.2f}   lr = {scheduler.get_last_lr()[0]:.4f}")
    if i % 1000 == 0:
        print(f"Iteration {i:4d}: loss = {loss.item():.2f}   lr = {scheduler.get_last_lr()[0]:.4f}")
        fname = os.path.join(OUT_DIR, f"{result_prefix}_at_iteration_{i}A.png")
        save_png(Image.fromarray(deprocess_image(combination_image)), fname)

