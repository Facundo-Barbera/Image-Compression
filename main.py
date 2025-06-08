import os
from src.compress_utility import (
    load_image,
    fast_fourier_transform,
    mask,
    inverse_fast_fourier_transform,
    save_image
)

def compress_image(input_path, output_path, compression_ratio=0.1):
    image_array = load_image(input_path)
    fourier_transformed = fast_fourier_transform(image_array)
    compressed = mask(fourier_transformed, ratio=compression_ratio)
    compressed_image = inverse_fast_fourier_transform(compressed)
    save_image(compressed_image, output_path)
    print(f"Compressed image saved to: {output_path}")

if __name__ == "__main__":
    input_dir = "images/raw"
    output_dir = "images/compressed"
    os.makedirs(output_dir, exist_ok=True)

    test_image = f"{input_dir}/test.ARW"
    output_image = f"{output_dir}/test_compressed.jpg"

    compress_image(test_image, output_image)

    print("Test completed successfully!")
