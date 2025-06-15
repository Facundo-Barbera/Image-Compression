import os
from pathlib import Path
from src import Image

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


def compress_with_image_class(input_path, output_path, compression_ratio=0.1, greyscale=False):
	img = Image(input_path)

	img.compress(output_path, ratio=compression_ratio, greyscale=greyscale)

	print(f"Compressed image saved to: {output_path}")


if __name__ == "__main__":
	input_dir = "images/raw"
	output_dir = "images/compressed"
	output_color_dir = "images/compressed_color"

	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(output_color_dir, exist_ok=True)

	for file in Path(input_dir).iterdir():
		if not file.is_file():
			continue
		if file.suffix.lower() not in IMAGE_EXTS:
			continue

		out_suffix = '.jpg' if file.suffix.lower() in RAW_EXTS else file.suffix.lower()
		out_file = f"{output_dir}/{file.stem}_compressed{out_suffix}"
		out_file_color = f"{output_color_dir}/{file.stem}_compressed{out_suffix}"

		try:
			compress_with_image_class(str(file), str(out_file), compression_ratio=0.01, greyscale=True)

			compress_with_image_class(str(file), str(out_file_color), compression_ratio=0.01, greyscale=False)
		except Exception as e:
			print(f"Error processing {file.name}: {e}")
			continue

	print("Test completed successfully!")