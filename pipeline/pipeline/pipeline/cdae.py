import torch
import torch.nn as nn
import numpy as np

class CDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),  nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def add_noise(flux: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Inject synthetic Gaussian noise for training."""
    return flux + np.random.normal(0, noise_level, flux.shape)


def train_cdae(clean_flux: np.ndarray, epochs: int = 50,
               lr: float = 1e-3, seq_len: int = 200):
    """
    Train CDAE on a single light curve via sliding window.
    clean_flux: normalized 1D numpy array
    Returns trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Sliding window segments
    segments = []
    for i in range(0, len(clean_flux) - seq_len, seq_len // 2):
        seg = clean_flux[i:i+seq_len]
        if len(seg) == seq_len:
            segments.append(seg)

    clean_t = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Training CDAE on {len(segments)} segments, device={device}")
    model.train()
    for epoch in range(epochs):
        noisy_t = clean_t + torch.randn_like(clean_t) * 0.02
        output = model(noisy_t)
        loss = loss_fn(output, clean_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    return model


def denoise(model: CDAE, flux: np.ndarray, seq_len: int = 200) -> np.ndarray:
    """
    Run trained CDAE over full light curve via sliding window.
    Returns denoised flux array of same length.
    """
    device = next(model.parameters()).device
    model.eval()
    output = np.zeros(len(flux))
    counts = np.zeros(len(flux))

    with torch.no_grad():
        for i in range(0, len(flux) - seq_len, seq_len // 4):
            seg = flux[i:i+seq_len]
            if len(seg) < seq_len:
                break
            t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(t).squeeze().cpu().numpy()
            output[i:i+seq_len] += pred
            counts[i:i+seq_len] += 1

    counts[counts == 0] = 1
    return output / counts


def compute_snr(raw: np.ndarray, denoised: np.ndarray) -> dict:
    """Compare SNR before and after denoising."""
    noise_raw = np.std(raw - np.median(raw))
    signal = np.abs(np.min(denoised) - np.median(denoised))
    noise_clean = np.std(denoised - np.median(denoised))
    snr_before = signal / noise_raw if noise_raw > 0 else 0
    snr_after  = signal / noise_clean if noise_clean > 0 else 0
    return {
        "snr_before": round(snr_before, 2),
        "snr_after":  round(snr_after, 2),
        "improvement": round(snr_after / snr_before, 2) if snr_before > 0 else 0
    }
