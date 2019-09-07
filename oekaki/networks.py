import os
import random
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

load_size = 286
# 286×286 → 256×256に
fine_size = 256
batch_size = 1
num_epoch = 200

lr = 0.0002
beta1 = 0.5
save_epoch_freq = 5
log_dir = 'logs'

class ResNetBlock(nn.Module):
    
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        ]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            
            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )
        
        self.model.apply(self._init_weights)
        
    def forward(self, input):
        return self.model(input)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model.apply(self._init_weights)
        
    def forward(self, input):
        return self.model(input)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class ImagePool():
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            
    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

class GANLoss(nn.Module):
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size())
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.zeros(input.size())
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor
    
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
                
class CycleGAN(object):
    
    def __init__(self, log_dir='logs'):
        self.G_X = Generator()
        self.G_Y = Generator()
        self.D_X = Discriminator()
        self.D_Y = Discriminator()
        
        # imageをpoolの中へ
        self.fake_X_pool = ImagePool(50)
        self.fake_Y_pool = ImagePool(50)
        
        # targetがfakeかrealかで変わるためGANLossクラスを作る
        self.criterionGAN = GANLoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        
        # Generatorは2つのパラメータを同時に更新
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_X.parameters(), self.G_Y.parameters()),
            lr=lr,
            betas=(beta1, 0.999))
        self.optimizer_D_X = torch.optim.Adam(self.D_X.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D_Y = torch.optim.Adam(self.D_Y.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_X)
        self.optimizers.append(self.optimizer_D_Y)
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
    def set_input(self, input):
        input_X = input['A']
        input_Y = input['B']
        
        self.input_X = input_X
        self.input_Y = input_Y
        self.image_paths = input['path_A']
        
    def backward_G(self, real_X, real_Y):
        lambda_idt = 0.5
        lambda_X = 10.0
        lambda_Y = 10.0
        
        # G_X, G_Yは変換先ドメインの本物画像を入力したときはそのまま出力するべき
        # netG_XはドメインXの画像からドメインYの画像を生成するGeneratorだが
        # ドメインYの画像も入れることができる
        # その場合は何も変換してほしくないという制約
        idt_X = self.G_X(real_Y)
        loss_idt_X = self.criterionIdt(idt_X, real_Y) * lambda_Y * lambda_idt
        
        idt_Y = self.G_Y(real_X)
        loss_idt_Y = self.criterionIdt(idt_Y, real_X) * lambda_X * lambda_idt
        
        # GAN loss D_X(G_X(X))
        fake_Y = self.G_X(real_X)
        pred_fake = self.D_X(fake_Y)
        loss_G_X = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_Y(G_Y(Y))
        fake_X = self.G_Y(real_Y)
        pred_fake = self.D_X(fake_X)
        loss_G_Y = self.criterionGAN(pred_fake, True)
        
        # forward cycle loss
        # real_X => fake_X => rec_Xが元のreal_Xに近いほどよい
        rec_X = self.G_Y(fake_Y)
        loss_cycle_X = self.criterionCycle(rec_X, real_X) * lambda_X
        
        # backward cycle loss
        # real_Y => fake_X => rec_Yが元のreal_Yに近いほど良い
        rec_Y = self.G_X(fake_X)
        loss_cycle_Y = self.criterionCycle(rec_Y, real_Y) * lambda_Y
        
        loss_G = loss_G_X + loss_G_Y + loss_cycle_X + loss_cycle_Y + loss_idt_X + loss_idt_Y
        loss_G.backward()
        
        loss_list = [loss_G_X.item(), loss_G_Y.item(), loss_cycle_X.item(), loss_cycle_Y.item(), loss_idt_X.item(), loss_idt_Y.item()]
        
        return loss_list, fake_X, fake_Y
    
    def backward_D_X(self, real_Y, fake_Y):
        # ドメインXから生成したfake_Yが本物か偽物かを見分ける
        
        # fake_Yの画像をpoolから選択
        fake_Y = self.fake_Y_pool.query(fake_Y)
        
        pred_real = self.D_X(real_Y)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # ドメインXから生成した偽物画像を入れたときは偽物になるように
        # Generatorまで勾配が伝播しないようにdetach()する
        pred_fake = self.D_X(fake_Y.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D_X = (loss_D_real + loss_D_fake) * 0.5
        loss_D_X.backward()
        
        return loss_D_X.item()
    
    def backward_D_Y(self, real_X, fake_X):
        # ドメインYから生成したfake_Xが本物か偽物かを見分ける
        
        # fake_Xの画像をpoolから選択
        fake_X = self.fake_X_pool.query(fake_X)
        
        pred_real = self.D_Y(real_X)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # 偽物画像は偽物と判断できるように
        pred_fake = self.D_Y(fake_X.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D_Y = (loss_D_real + loss_D_fake) * 0.5
        loss_D_Y.backward()
        
        return loss_D_Y.item()
    
    def optimize(self):
        real_X = self.input_X
        real_Y = self.input_Y
        
        # Generatorを更新
        self.optimizer_G.zero_grad()
        loss_list, fake_X, fake_Y = self.backward_G(real_X, real_Y)
        loss_G_X = loss_list[0]
        loss_G_Y = loss_list[1]
        loss_cycle_X = loss_list[2]
        loss_cycle_Y = loss_list[3]
        loss_idt_X = loss_list[4]
        loss_idt_Y = loss_list[5]
        self.optimizer_G.step()
        
        # D_Xを更新
        self.optimizer_D_X.zero_grad()
        loss_D_X = self.backward_D_X(real_Y, fake_Y)
        self.optimizer_D_X.step()
        
        # D_Yを更新
        self.optimizer_D_Y.zero_grad()
        loss_D_Y = self.backward_D_Y(real_X, fake_X)
        self.optimizer_D_Y.step()
        
        ret_loss = [loss_G_X, loss_D_X,
                    loss_G_Y, loss_D_Y,
                    loss_cycle_X, loss_cycle_Y,
                    loss_idt_X, loss_idt_Y]
        
        return np.array(ret_loss)

    def train(self, data_loader):
        
        self.G_X.train()
        self.G_Y.train()
        self.D_X.train()
        self.D_Y.train()
        
        running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for batch_idx, data in enumerate(data_loader):
            self.set_input(data)
            losses = self.optimize()
            running_loss += losses
        running_loss /= len(data_loader)
        return running_loss
    
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.log_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
    
    def load_network(self, network, network_label, epoch_label):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.log_dir, load_filename)
        network.load_state_dict(torch.load(load_path))

        
    def save(self, label):
        self.save_network(self.G_X, 'G_X', label)
        self.save_network(self.D_X, 'D_X', label)
        self.save_network(self.G_Y, 'G_Y', label)
        self.save_network(self.D_Y, 'D_Y', label)
        
    def load(self, label):
        self.load_network(self.G_X, 'G_X', label)
        self.load_network(self.D_X, 'D_X', label)
        self.load_network(self.G_Y, 'G_Y', label)
        self.load_network(self.D_Y, 'D_Y', label)