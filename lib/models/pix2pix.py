"""
Create by Fabio Hellmann - based on Erik Linder-Noren's implementation:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
"""

import os.path

import pytorch_lightning
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from lib.datasets.image_dataset import ImageDataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_a, img_b):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_a, img_b), 1)
        return self.model(img_input)


##############################
#        Lightning
##############################


class Pix2Pix(pytorch_lightning.LightningModule):
    def __init__(self, data_dir: str, models_dir: str, output_dir: str, n_epochs: int,
                 dataset_name: str, batch_size: int, lr: float, b1: float, b2: float, n_cpu: int,
                 img_size: int, device: int):
        """
        Create a Pix2Pix Network.
        @param data_dir: The directory of the data.
        @param models_dir: The directory of the models.
        @param output_dir: The directory of the output.
        @param n_epochs: The number of epochs.
        @param dataset_name: The name of the dataset which is appended to the output_dir.
        @param batch_size: The size of the batches to process.
        @param lr: The learning rate.
        @param b1: The beta 1 value for the optimizer.
        @param b2: The beta 2 value for the optimizer.
        @param n_cpu: The number of cpus.
        @param img_size: The size of the image.
        @param device: The device to run the computation on.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.n_epochs = n_epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.n_cpu = n_cpu
        self.img_size = img_size

        self.transforms_ = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_GAN.to(device)
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.criterion_pixelwise.to(device)
        self.lambda_pixel = 100
        self.generator = GeneratorUNet()
        self.generator.to(device)
        self.discriminator = Discriminator()
        self.discriminator.to(device)

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()
        # Model inputs
        real_a = batch["B"]
        real_b = batch["A"]
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, self.img_size // 2 ** 4, self.img_size // 2 ** 4)
        # Adversarial ground truths
        valid = torch.ones((real_a.size(0), *patch), requires_grad=False).to(self.device)
        fake = torch.zeros((real_a.size(0), *patch), requires_grad=False).to(self.device)
        # ------------------
        #  Train Generators
        # ------------------
        self.toggle_optimizer(optimizer_g)
        # GAN loss
        fake_b = self.generator(real_a)
        pred_fake = self.discriminator(fake_b, real_a)
        loss_gan = self.criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = self.criterion_pixelwise(fake_b, real_b)
        # Total loss
        loss_g = loss_gan + self.lambda_pixel * loss_pixel
        self.log('G loss', loss_g, prog_bar=True)
        self.log('G pixel', loss_pixel, prog_bar=True)
        self.log('G adv', loss_gan, prog_bar=True)
        self.manual_backward(loss_g)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.toggle_optimizer(optimizer_d)
        # Real loss
        pred_real = self.discriminator(real_b, real_a)
        loss_real = self.criterion_GAN(pred_real, valid)
        # Fake loss
        fake_b = self.generator(real_a)
        pred_fake = self.discriminator(fake_b.detach(), real_a)
        loss_fake = self.criterion_GAN(pred_fake, fake)
        # Total loss
        loss_d = 0.5 * (loss_real + loss_fake)
        self.log('D loss', loss_d, prog_bar=True)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            real_a = batch["B"]
            real_b = batch["A"]
            fake_b = self.generator(real_a)
            img_sample = torch.cat((real_b.data, real_a.data, fake_b.data), -2)
            save_image(img_sample,
                       os.path.join(self.out_dir, f'{self.current_epoch}-{self.global_step}.png'),
                       nrow=5, normalize=True)
            grid = make_grid(img_sample, nrow=5, normalize=True)
            self.logger.experiment.add_image('images', grid, self.global_step)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr,
                                       betas=(self.b1, self.b2))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr,
                                       betas=(self.b1, self.b2))
        return [optimizer_g, optimizer_d], []

    def train_dataloader(self):
        return DataLoader(
            ImageDataset(os.path.join(self.data_dir, self.dataset_name),
                         transforms_=self.transforms_),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )

    def val_dataloader(self):
        return DataLoader(
            ImageDataset(os.path.join(self.data_dir, self.dataset_name),
                         transforms_=self.transforms_, mode="val"),
            batch_size=10,
            shuffle=True,
            num_workers=self.n_cpu,
        )
