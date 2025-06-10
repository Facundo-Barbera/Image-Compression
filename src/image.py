import numpy as np
import rawpy
import os
from PIL import (
    Image as PILImage,
    UnidentifiedImageError
)


class Image:
    
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.image_array = None
        
        if image_path:
            self.image_array = self._load_image(image_path)
    
    def _load_image(self, image_path, greyscale=False):
        img = None
        
        try:
            img = PILImage.open(image_path)
        except (UnidentifiedImageError, OSError):
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(
                    no_auto_bright=True,
                    output_bps=8
                )
                img = PILImage.fromarray(rgb)
        
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        if greyscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
            
        image_array = np.array(img, dtype=float)
        return image_array
    
    def _fast_fourier_transform(self, image_array):
        if len(image_array.shape) == 2:
            fourier_transformed = np.fft.fft2(image_array)
            fourier_transformed_shifted = np.fft.fftshift(fourier_transformed)
            return fourier_transformed_shifted
        else:
            height, width, channels = image_array.shape
            fourier_transformed_shifted = np.zeros((height, width, channels), dtype=complex)
            
            for c in range(channels):
                fourier_transformed = np.fft.fft2(image_array[:, :, c])
                fourier_transformed_shifted[:, :, c] = np.fft.fftshift(fourier_transformed)
            
            return fourier_transformed_shifted
    
    def _mask(self, fourier_transformed, ratio=0.5):
        if len(fourier_transformed.shape) == 2:
            magnitude = np.abs(fourier_transformed)
            threshold = np.percentile(magnitude, 100 * (1 - ratio))
            
            mask = magnitude > threshold
            fourier_transformed_compressed = fourier_transformed * mask
            
            return fourier_transformed_compressed
        else:
            height, width, channels = fourier_transformed.shape
            fourier_transformed_compressed = np.zeros_like(fourier_transformed)
            
            for c in range(channels):
                magnitude = np.abs(fourier_transformed[:, :, c])
                threshold = np.percentile(magnitude, 100 * (1 - ratio))
                
                channel_mask = magnitude > threshold
                fourier_transformed_compressed[:, :, c] = fourier_transformed[:, :, c] * channel_mask
            
            return fourier_transformed_compressed
    
    def _inverse_fast_fourier_transform(self, fourier_transformed):
        if len(fourier_transformed.shape) == 2:
            img_back = np.fft.ifft2(np.fft.ifftshift(fourier_transformed)).real
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            
            return img_back
        else:
            height, width, channels = fourier_transformed.shape
            img_back = np.zeros((height, width, channels), dtype=np.uint8)
            
            for c in range(channels):
                channel_back = np.fft.ifft2(np.fft.ifftshift(fourier_transformed[:, :, c])).real
                img_back[:, :, c] = np.clip(channel_back, 0, 255).astype(np.uint8)
            
            return img_back
    
    def _save_image(self, image_array, output_path):
        img = PILImage.fromarray(image_array)

        ext = os.path.splitext(output_path)[1].lower()
        save_kwargs = {}

        if ext in ('.jpg', '.jpeg'):
            # JPEG: set maximum quality, no subsampling
            save_kwargs['quality']     = 100
            save_kwargs['subsampling'] = 0
        elif ext == '.png':
            # PNG: no compression
            save_kwargs['compress_level'] = 0

        img.save(output_path, **save_kwargs)
    
    def compress(self, output_path, ratio=0.5, greyscale=False):
        if self.image_array is None or greyscale != (len(self.image_array.shape) == 2):
            self.image_array = self._load_image(self.image_path, greyscale)

        fourier_transformed = self._fast_fourier_transform(self.image_array)
        fourier_transformed_compressed = self._mask(fourier_transformed, ratio)
        compressed_image = self._inverse_fast_fourier_transform(fourier_transformed_compressed)

        self._save_image(compressed_image, output_path)
        
        return compressed_image