import numpy as np
import rawpy
from PIL import (
	Image,
	UnidentifiedImageError
)


def load_image(image_path: str):
	img = None

	try:
		img = Image.open(image_path)
	except (UnidentifiedImageError, OSError):
		with rawpy.imread(image_path) as raw:
			rgb = raw.postprocess(
				no_auto_bright=True,
				output_bps=8
			)
			img = Image.fromarray(rgb)

	if img is None:
		raise ValueError(f"Could not load image from {image_path}")

	img = img.convert('L')
	image_array = np.array(img, dtype=float)
	return image_array


def fast_fourier_transform(image_array: np.ndarray):
	# Perform the Fast Fourier Transform
	fourier_transformed = np.fft.fft2(image_array)

	# Shift the zero frequency component to the center
	fourier_transformed_shifted = np.fft.fftshift(fourier_transformed)

	# Return the transformed array
	return fourier_transformed_shifted


def mask(fourier_transformed: np.ndarray, ratio: float = 0.1):
	magnitude = np.abs(fourier_transformed)
	threshold = np.percentile(magnitude, 100 * (1 - ratio))

	# Create a mask where the magnitude is above the threshold
	mask = magnitude > threshold
	fourier_transformed_compressed = fourier_transformed * mask

	# Return the compressed Fourier transformed array
	return fourier_transformed_compressed


def inverse_fast_fourier_transform(fourier_transformed: np.ndarray):
	# Inverse shift the zero frequency component back to the original
	img_back = np.fft.ifft2(np.fft.ifftshift(fourier_transformed)).real
	img_back = np.clip(img_back, 0, 255).astype(np.uint8)

	# Return the reconstructed image
	return img_back


def save_image(image_array: np.ndarray, output_path: str):
	# Convert the array to an image
	img = Image.fromarray(image_array)

	# Save the image
	img.save(output_path)
