import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Resampling


def dither_image(image_path) -> np.ndarray:
    # Load the image
    img_array = load_image(image_path)

    # Define the size of the image
    height, width = img_array.shape

    # Create an output array
    dithered_array = np.zeros_like(img_array)

    # Define a simple error diffusion matrix (Floyd-Steinberg dithering)
    error_diffusion_matrix = [
        [0, 0, 7 / 16],
        [3 / 16, 5 / 16, 1 / 16]
    ]

    # Apply dithering algorithm
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = 255 * (old_pixel // 128)
            dithered_array[y, x] = new_pixel
            quantization_error = old_pixel - new_pixel

            # Distribute the quantization error to neighboring pixels
            for dy in range(2):
                for dx in range(3):
                    ny = y + dy
                    nx = x + dx - 1
                    if 0 <= ny < height and 0 <= nx < width:
                        img_array[ny, nx] = min(255, max(0, img_array[ny, nx] + quantization_error * error_diffusion_matrix[dy][dx]))

    return dithered_array


def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    return img_array


def save_image_from_array(dithered_array: np.ndarray, output_path: str):
    dithered_image = Image.fromarray(dithered_array)
    dithered_image.save(output_path)


def resize_image_array(image_array: np.ndarray, max_size):
    # Get original dimensions
    original_height, original_width = image_array.shape

    # Determine the scaling factor while maintaining the aspect ratio
    if original_height > original_width:
        scale_factor = max_size / original_height
    else:
        scale_factor = max_size / original_width

    # Calculate new dimensions
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    # Create an empty array for the new image
    resized_img = np.zeros((new_height, new_width), dtype=image_array.dtype)

    # Calculate the ratio of old dimensions to new dimensions
    height_ratio = original_height / new_height
    width_ratio = original_width / new_width

    # Populate the resized image
    for i in range(new_height):
        for j in range(new_width):
            # Map the coordinates of the new image to the coordinates of the old image
            old_i = int(i * height_ratio)
            old_j = int(j * width_ratio)
            resized_img[i, j] = image_array[old_i, old_j]

    return resized_img


def resize_and_center_image(image_path, output_path):
    # Open the original image
    img = Image.open(image_path)

    # Calculate the new size maintaining the aspect ratio
    max_size = (400, 400)
    img.thumbnail(max_size, resample=Resampling.LANCZOS)

    # Create a new black image with size 400x400
    new_img = Image.new("RGB", (400, 400), "black")

    # Calculate the position to paste the resized image onto the new image
    paste_position = ((400 - img.width) // 2, (400 - img.height) // 2)

    # Paste the resized image onto the new image
    new_img.paste(img, paste_position)

    # Save the final image
    new_img.save(output_path)


if __name__ == '__main__':
    # dither_and_save('images/Bernardo_Houssay.jpeg', 'images/Houssay.jpg')
    # dither_and_save('images/270px-Luis_Federico_Leloir_-_young.jpg', 'images/Leloir.jpg')
    # dither_and_save('images/Cesar-Milstein400.png', 'images/Milstein.jpg')

    resize_and_center_image('images/Bernardo_Houssay.jpeg', 'images/resized/Houssay.png')
    resize_and_center_image('images/Luis_Federico_Leloir.jpg', 'images/resized/Leloir.png')
    resize_and_center_image('images/Cesar-Milstein400.png', 'images/resized/Milstein.png')