import torch
import numpy as np
from PIL import Image
import seaborn as sns
from matplotlib import pyplot as plt

from ptych import solve_inverse, PtychStudy
from ptych.core.zernike import precompute_zernike_basis, make_zernike_pupil, ZernikeParams

from pathlib import Path
import datetime
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import json


# Generate path to data parent

srcInitialParent = Path.home() / "Desktop" / "fpmData"   # modify as desired for data destination

# Load true high-quality image and convert to grayscale float [0, 1]

app = QApplication(sys.argv)   # ✅ create app first
win = QMainWindow()
#win.show()

srcDirPath = QFileDialog.getExistingDirectory(
    win,
    'Select study folder',
    str(srcInitialParent),
)

BASE_DIR = Path(srcDirPath)
print('BASE_DIR:', BASE_DIR)

study = PtychStudy.from_disk(BASE_DIR)
eps = 1e-8

# Initialize object and pupil with upsampled dimensions
upsample_ratio = 4
dims = study.manifest.capture_dimensions
csize = study.captures.shape
print('Loaded', csize[0], 'raw data images of size', csize[1], 'x', csize[2])

init_amp = torch.nn.functional.interpolate(
    study.captures[0:1, :, :].unsqueeze(1),  # [B, n, n]
    scale_factor=upsample_ratio,
    mode='bilinear'
).squeeze()


if True:
    cCnt = study.captures.shape[0]
    xCnt = np.ceil(np.sqrt(cCnt)).astype(int)
    yCnt = np.ceil(cCnt / xCnt).astype(int)
    print(cCnt, xCnt, yCnt)
    fig, axes = plt.subplots(xCnt, yCnt, figsize=(10, 8))
    i = 0
    for x in range(xCnt):
        for y in range(yCnt):
            amplitude = study.captures[i,:,:]
            axes[x, y].imshow(amplitude, cmap='gray')
            axes[x, y].set_title(f"Image {i+1}")
            axes[x, y].axis("off")
            i = i + 1
            if i >= cCnt:
                break
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.05,
        wspace=0.0,
        hspace=0.02
    )
    plt.show(block=True)




init_amp = torch.sqrt(init_amp + eps)  # Convert intensity to amplitude
init_phase = torch.zeros_like(init_amp)  # or small random noise

object_tensor = init_amp * torch.exp(1j * init_phase)


# Initialize pupil using Zernike basis

N = dims.height * upsample_ratio
basis = precompute_zernike_basis(N, num_phase_terms=3, num_amp_terms=3)
phase_coeffs = torch.zeros(basis.num_phase_terms)
amp_coeffs = torch.zeros(basis.num_amp_terms)
amp_coeffs[0] = 1.0  # Piston = uniform amplitude
rad_fraction = torch.tensor(0.15)  # Learnable radius fraction

pupil = ZernikeParams(phase_coeffs, amp_coeffs, basis, rad_fraction)

object, pupil, metrics = solve_inverse(
    study.captures,
    object_tensor,
    pupil,
    study.kx_batch,
    study.ky_batch,
    torch_device="mps",
)

# Save object result as PNG
object_amplitude: np.ndarray[tuple[int, int], np.dtype[np.float32]] = object.abs().cpu().numpy()
# Normalize to 0-255 range
object_amplitude_u8 = np.asarray(
    object_amplitude / object_amplitude.max() * 255, dtype=np.uint8
)
Image.fromarray(object_amplitude_u8).save(f"{BASE_DIR}/object_result.png")




# Save pupil result as PNG
assert isinstance(pupil, ZernikeParams)
pupil_tensor = make_zernike_pupil(pupil.phase_coeffs, pupil.amp_coeffs, pupil.basis, pupil.rad_fraction)
print(f"Learned rad_fraction: {pupil.rad_fraction.item():.6f}")
pupil_amplitude = pupil_tensor.abs().cpu().numpy()
pupil_amplitude_u8 = np.asarray(
    pupil_amplitude / pupil_amplitude.max() * 255, dtype=np.uint8
)
Image.fromarray(pupil_amplitude_u8).save(f"{BASE_DIR}/pupil_result.png")

# Plot and save metrics
sns.set_theme(style="darkgrid")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True) # pyright: ignore[reportAny]

epochs = range(len(metrics['loss']))

sns.lineplot(x=list(epochs), y=metrics['loss'], ax=ax1) # pyright: ignore[reportAny]
ax1.set_ylabel('Loss') # pyright: ignore[reportAny]
ax1.set_title('Training Metrics') # pyright: ignore[reportAny]

sns.lineplot(x=list(epochs), y=np.log(metrics['loss']), ax=ax2) # pyright: ignore[reportAny]
ax2.set_ylabel('Log Loss') # pyright: ignore[reportAny]

plt.tight_layout()
plt.savefig(f"{BASE_DIR}/metrics.png", dpi=150)
plt.close()

#if True:
#    plt.show(object_amplitude_u8, cmap='gray')
#    plt.set_title("Result")
#    plt.axis("off")
#    plt.tight_layout()
#    plt.show(block=True)
