import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion():
    # Diffusion for lyric to music generation
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, melody_len=20, device="cuda"):

        # Store params
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.melody_len = melody_len
        self.device = device

        # Calculate alpha hats for closed form noise usage
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_melodies(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_t.float(), eps.float()

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, 1, self.melody_len)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
'''
class UNet_1D(nn.Module):
    # UNet architecture
    def __init__(self):
        super(UNet_1D, self).__init__()

        # Input spectrogram shape = (batch, 3, 1, 20) = (batch, channel, height, song)

        # Encoder conv blocks (2)
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1,3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )

        # Transformer blocks (2)
        self.t1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )

        # Decoder deconv blocks(4)
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(1,3), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, kernel_size=(1,3), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):

        print('PRE-ENCODER SHAPE:', x.shape)

        # Convolutional encoder
        x = self.e1(x)

        print('E1 SHAPE:', x.shape)

        x = self.e2(x)

        print('E2 SHAPE:', x.shape)

        # Convolutional transformer
        x = self.t1(x)
        x = self.t2(x)

        # Convolutional decoder
        x = self.d1(x)
        x = self.d2(x)

        return x
'''