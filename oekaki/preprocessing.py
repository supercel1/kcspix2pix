from PIL import Image

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

def make_input(img_path, img_name):
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