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

    def noise_melodies_1d(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_t.float(), eps.float()

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, 1, self.melody_len)).to(self.device)
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

    def sample_1d(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.melody_len)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
    
    def sample_1d_wText(self, model, y, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.melody_len)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, y)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x

class LSTMNet(nn.Module):

    # LSTM architecture
    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        
        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pos_emb_size = input_size + 1

        # Params for RNN
        hidden_size = 200
        num_layers = 2
        dropout = 0.2

        # Bidirectional encoder to get globally aware encodings
        # at each input
        self.rnn = nn.LSTM(input_size + pos_emb_size, hidden_size, 
                           num_layers, bidirectional=True, 
                           batch_first=True, dropout=dropout)

        # FC layer to classify the noise
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):

        # Get positional encoding of timestep t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.input_size + 1)
        #print('init', t.shape)
        x = x.transpose(2, 1)
        t = t.unsqueeze(1).repeat(1, x.shape[1], 1)

        #print('Shapes:', x.shape, t.shape)

        # Concatenate positional encoding with noisy x
        x = torch.cat((x, t), dim=-1)

        #print('Concat shape:', x.shape)

        # Pass through LSTM layers
        x, _ = self.rnn(x)

        #print('LSTM shape:', x.shape)

        # Pass through FC to classify noise
        x = self.fc(x)

        x = x.transpose(2, 1)
        #print('FC Shape:', x.shape)

        return x

class LSTMNet_wText(nn.Module):

    # LSTM architecture
    def __init__(self, input_size):
        super(LSTMNet_wText, self).__init__()
        
        self.input_size = input_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pos_emb_size = input_size + 1

        # Params for RNN
        hidden_size = 200
        num_layers = 2
        dropout = 0.2
        text_emb_size = 3

        # Bidirectional encoder to get globally aware encodings
        # at each input
        self.rnn = nn.LSTM(input_size + pos_emb_size + text_emb_size, hidden_size, 
                           num_layers, bidirectional=True, 
                           batch_first=True, dropout=dropout)

        # FC layer to classify the noise
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
        )

        # Text emb FC
        self.textfc = nn.Linear(20, text_emb_size)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, text_embs):

        # Get positional encoding of timestep t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.input_size + 1)

        x = x.transpose(2, 1)
        t = t.unsqueeze(1).repeat(1, x.shape[1], 1)

        text_embs = text_embs.reshape(text_embs.shape[0], 20, 20).float().cuda()
        text_embs = self.textfc(text_embs)

        # Concatenate positional encoding with noisy x and syllable embedding
        x = torch.cat((x, t, text_embs), dim=-1)

        # Pass through LSTM layers
        x, _ = self.rnn(x)

        # Pass through FC to classify noise
        x = self.fc(x)

        x = x.transpose(2, 1)

        return x