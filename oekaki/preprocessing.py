from PIL import Image

import numpy as np

import torchvision.transforms as transforms

resize = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
class ImageTransform():

    def __init__(self, resize, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)

class ImageTransform_L():

    def __init__(self, resize, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize), Image.BICUBIC),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)

def make_input(img_path, img_name):
    img = Image.open(img_path)
    if img.mode == 'L':
        transforms_L = ImageTransform_L(resize, mean, std)
        img_transformed = transforms_L(img)
        img_input = img_transformed.unsqueeze_(0)
        return img_input
    else:
        img = to_RGB(Image.open(img_path), img_name)
        transform = ImageTransform(resize, mean, std)
        img_transformed = transform(img)
        img_input = img_transformed.unsqueeze_(0) # ミニバッチ分の次元を追加
        return img_input

def to_RGB(img, img_name):
    if img.mode == 'RGB':
        return img
    
    img.load()
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])

    file_name = img_name + '.jpg'
    background.save(file_name, 'JPEG', quality=80)
    return Image.open(file_name)

def to_PIL(img):
    img_np = img[0].detach().numpy()
    img_np = 0.5 * (img_np + 1)
    img_np = np.transpose(img_np, (1, 2, 0))

    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return img_pil