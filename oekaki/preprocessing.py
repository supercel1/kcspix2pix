from PIL import Image

import torchvision.transforms as transforms

resize = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
class ImageTransform():

    def __init__(self, resize, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transform(img)

def make_input(img_path):
    img = Image.open(img_path)

    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img)
    img_input = img_transformed.unsqueeze_(0) # ミニバッチ分の次元を追加
    return img_input