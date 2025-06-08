import os
from pathlib import Path

from src.compress_utility import (
    load_image,
    fast_fourier_transform,
    mask,
    inverse_fast_fourier_transform,
    save_image
)

RAW_EXTS = {
    ".arw",
    ".cr2",
    ".nef",
    ".dng",
    ".raf",
    ".rw2"
}
IMAGE_EXTS = RAW_EXTS | {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

def compress_image(input_path, output_path, compression_ratio=0.01):
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

    for file in Path(input_dir).iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        out_suffix = '.jpg' if file.suffix.lower() in RAW_EXTS else file.suffix.lower()
        out_file = f"{output_dir}/{file.stem}_compressed{out_suffix}"

        try:
            compress_image(str(file), str(out_file), compression_ratio=0.01)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue

    print("Test completed successfully!")
