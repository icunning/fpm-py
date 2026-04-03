"""Demo script for generating synthetic FPM captures from bars.png."""

import numpy as np
import torch
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study
from ptych.core.zernike import precompute_zernike_basis, make_zernike_pupil

from tkinter import Tk, filedialog
from pathlib import Path
import matplotlib.pyplot as plt

import json

from datetime import datetime
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from qtlib import select_roi, show_grayscale_subplots, MultiImageWindow


#if __name__ == "__main__":

# Start Qt for QtWidgets

app = QApplication(sys.argv)   # ✅ create app first
win = QMainWindow()
#win.show()


# Generate path to data parent

srcInitialParent = Path.home() / "Desktop" / "fpmData"   # modify as desired for data destination

# Load true high-quality image and convert to grayscale float [0, 1]

srcFilePath,_ = QFileDialog.getOpenFileName(
    win,
    caption='Select input image file',
    directory=str(srcInitialParent),
    filter='',
    initialFilter='',
    options=QFileDialog.Options()
)
srcDir = Path(srcFilePath).parent
print('Source:', srcDir)


# Get "truth" image and convert to grayscale float [0, 1]

img = Image.open(srcFilePath).convert("L")
amplitude = np.array(img, dtype=np.float32) / 255.0

# User select ROI

roi = select_roi(img)
print("Returned ROI:", roi)
if roi == None:
    exit(0)

# Force to even width and height
x = roi[0]
y = roi[1]
w = roi[2]
h = roi[3]
roi = [x, y, w-w%2, h-h%2]

print("Returned ROI:", roi)

amplitude = amplitude[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

# Generate datetime string, unique for this study

dateTimeString = datetime.now().strftime("%Y%m%d-%H%M%S")
print('Analysis dateTime stamp:', dateTimeString)


# Create object tensor: phase proportional to amplitude
# Scale phase to [0, 2*pi] range
phase = amplitude * 2 * np.pi
object_tensor = torch.from_numpy(amplitude * np.exp(1j * phase)).to(torch.complex64)

# Create pupil tensor using Zernike basis
N = object_tensor.shape[0]

# Precompute Zernike basis
basis = precompute_zernike_basis(N)

# Define Zernike coefficients
phase_coeffs = torch.zeros(basis.num_phase_terms)  # No aberrations
amp_coeffs = torch.zeros(basis.num_amp_terms)
amp_coeffs[0] = 1.0  # Piston = uniform amplitude
rad_fraction = 0.15  # rad_fraction=0.15 matches current 0.30/2 radius

# Generate pupil (use_softplus=False for exact amplitude)
pupil_tensor = make_zernike_pupil(phase_coeffs, amp_coeffs, basis, rad_fraction, use_softplus=False)

object_amplitude_u8 = np.asarray(
    pupil_tensor.real / pupil_tensor.real.max() * 255, dtype=np.uint8
)

#imOrig = np.fft.fftshift(object_amplitude_u8)

#plt.imshow(imOrig, cmap='gray')
#plt.axis("off")
#plt.show(block=True)


#result = Image.fromarray(object_amplitude_u8)
#.save(f"tmp/test/object_result.png")


# Generate coordinates of led array

def generate_concentric_coordinates(radii, positions_per_ring):
	"""
	Generate Cartesian coordinates for concentric rings, formatted to three decimal places.

	Args:
	radii (list of float): Radii of each ring.
	positions_per_ring (list of int): Number of positions in each ring.

	Returns:
	np.array: Array containing Cartesian coordinates to be used in json file.
	"""
	# Initialize the list to hold all positions
	all_positions = []

	# Iterate over each ring
	for i, radius in enumerate(radii):
		# Number of positions in the current ring
		num_positions = positions_per_ring[i]
		# Angle between each position
		angle_increment = 2 * np.pi / num_positions

		# Generate coordinates for each position in the ring
		for j in range(num_positions):
			angle = angle_increment * j
			x = round(radius * np.cos(angle), 6)
			y = round(radius * np.sin(angle), 6)
			#x=-x
			all_positions.append([x, y])
			r = np.sqrt(x**2 + y**2)
			print(f"{i} {j} ({x}, {y}) {round(r,3)}, {round(angle/2/np.pi * 360)}")

	# Convert list to a numpy array
	positions_array = np.array(all_positions)
	return all_positions#positions_array


radii = [0, 3e-3, 6e-3, 9.5e-3, 13e-3, 16.5e-3, 20.8e-3]
positions_per_ring = [1, 8, 12, 16, 24, 36, 48]
ledPositions = generate_concentric_coordinates(radii, positions_per_ring)


# Create output directory

resultsDir = Path(srcDir / str(dateTimeString + "synthetic"))

# Generate info.json

# Set downsample factor
downsample_ratio = 4
optical_magnification = 1.7
sensor_pixel_size = 1.12e-6
wavelengthGreen = 0.525e-6
z_height = 60.5e-3
original_width = int(roi[2] / downsample_ratio)#128
original_height = int(roi[3] / downsample_ratio)#128
obj = {
    "study_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "created_at": dateTimeString,
    "version": "1.0",
    "magnification": optical_magnification,
    "sensor_pixel_size": sensor_pixel_size,
    "capture_dimensions": {"width": original_width, "height": original_height},
    "captures": [],
    "metadata": {
        "sample": "test_target",
        "notes": "Synthetic test data for PtychStudy validation"
    }
}

f = 2  # REQUIRE ~2 TO GET HIGH-PASS IMAGES similar to experiment - ??????
for i, (x, y) in enumerate(ledPositions):
    xf = x * f
    yf = y * f
    obj["captures"].append({
        "filename": f"im_{i}.npy",
        "wavelength": wavelengthGreen,
        "led_positions": [{"x": f*x, "y": f*y, "z": z_height}]
    })

print(resultsDir)
resultsDir.mkdir(parents=True, exist_ok=True)

# Write info.json - it is used by generate_synthetic_study

with open(Path(resultsDir / "info.json"), "w") as f:
    json.dump(obj, f, indent=1)

# Run synthetic study generation using the new info.json

captures = generate_synthetic_study(
    dir_path=resultsDir,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor,
    downsample_ratio=downsample_ratio,
)


if True:
    n = len(captures)
    ncols = int(16)#np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    print(n, nrows, ncols)
    imageWin = show_grayscale_subplots(captures, nrows=nrows, ncols=ncols, titles=None)
    app.exec_()

sys.exit(0)
