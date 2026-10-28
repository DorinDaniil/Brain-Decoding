"""Network and inference model, self-contained (torch only).

This file is copied verbatim into the Codabench ``submission.py`` by
``emg_pose.export``, so it must not import anything from this project.

PoseNet: wrist sEMG window (B, 16, T) at 2 kHz, microvolts -> 20 joint angles
(B, 20, T) in radians. The whole window is available, so the network is
bidirectional and regresses angles directly (no velocity integration):

    log-spectrogram (50 Hz frames) -> per-window gain removal -> channel norm
    -> rotation-invariant MLP over circular electrode shifts -> conv block
    -> bidirectional GRU -> linear head + mean pose -> linear upsampling to T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationInvariantMLP(nn.Module):
    """Per frame: the same MLP on the electrode ring rolled by each offset, then averaged.

    The 16 electrodes sit on a ring around the wrist, so a band put on slightly
    rotated is a circular shift of the channels.
    """

    def __init__(self, d_in, d_out, offsets=(-1, 0, 1)):
        super().__init__()
        self.offsets = tuple(offsets)
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out), nn.ReLU())

    def forward(self, x):  # x (B, F, C, K): frames, channels, frequency bins
        rolled = torch.stack([x.roll(o, dims=2) for o in self.offsets], dim=2)  # (B, F, R, C, K)
        return self.mlp(rolled.flatten(3)).mean(dim=2)


class PoseNet(nn.Module):
    def __init__(self, n_chans=16, n_joints=20, sfreq=2000.0, frame_rate=50, n_fft=64,
                 hidden=384, rnn=256, layers=2, dropout=0.2, mean_pose=None):
        super().__init__()
        self.config = dict(n_chans=n_chans, n_joints=n_joints, sfreq=float(sfreq), frame_rate=frame_rate,
                           n_fft=n_fft, hidden=hidden, rnn=rnn, layers=layers, dropout=dropout)
        self.hop = int(round(sfreq / frame_rate))
        self.n_fft = n_fft
        n_bins = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(n_fft))
        self.norm = nn.BatchNorm2d(n_chans)  # per electrode, statistics over frequency and time
        self.rotation = RotationInvariantMLP(n_chans * n_bins, hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.gru = nn.GRU(hidden, rnn, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * rnn, n_joints)
        mean_pose = torch.zeros(n_joints) if mean_pose is None else torch.as_tensor(mean_pose, dtype=torch.float32)
        self.register_buffer("mean_pose", mean_pose.clone())

    def spectrogram(self, X):
        """(B, C, T) microvolts -> log power (B, C, K, F), with the window's overall gain removed."""
        B, C, T = X.shape
        spec = torch.stft(X.reshape(B * C, T).float(), self.n_fft, self.hop, window=self.window,
                          center=True, return_complex=True)
        logp = torch.log(spec.abs().pow(2) + 1e-6).reshape(B, C, *spec.shape[-2:])
        return logp - logp.mean(dim=(1, 2, 3), keepdim=True)  # same gain on all electrodes -> invariant

    def forward(self, X):
        """(B, C, T) -> angles at the frame rate (B, J, F)."""
        h = self.norm(self.spectrogram(X))                    # (B, C, K, F)
        h = self.rotation(h.permute(0, 3, 1, 2))              # (B, F, hidden)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return (self.head(h) + self.mean_pose).transpose(1, 2)

    def angles(self, X):
        """(B, C, T) -> angles at the input rate (B, J, T), linearly interpolated from the frames."""
        return F.interpolate(self(X), size=X.shape[-1], mode="linear", align_corners=True)


class PoseEnsemble(nn.Module):
    """Average of PoseNets, optionally over circular shifts of the electrodes too. Stateless."""

    def __init__(self, nets, tta_offsets=(0,)):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.tta_offsets = tuple(tta_offsets)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.nets[0].mean_pose.device)
        outs = [net.angles(X.roll(o, dims=1)) for net in self.nets for o in self.tta_offsets]
        return torch.stack(outs).mean(0)
