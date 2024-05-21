import numpy as np
from tqdm import tqdm
from image_loader import dither_image, save_image_from_array, resize_image_array
import torch


# class HopfieldNetwork:
#     def __init__(self, size):
#         self.size = size
#         self.weights = np.zeros((size, size))
#
#     def train(self, data):
#         for pattern in data:
#             print(f"Pattern length: {len(pattern)}")
#             self.weights += np.outer(pattern, pattern)
#         np.fill_diagonal(self.weights, 0)
#
#     def predict(self, pattern, iterations: int = 100):
#         result = np.copy(pattern)
#         for _ in tqdm(range(iterations)):
#             for i in range(self.size):
#                 result[i] = 1 if np.dot(self.weights[i], result) > 0 else -1
#         return result


class HopfieldNetworkTorch:
    def __init__(self, size, force_cpu=False):
        self.size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() and force_cpu is False else "cpu")
        self.weights = torch.zeros((size, size), device=self.device, dtype=torch.bfloat16)

    def train(self, data):
        for pattern in tqdm(data):
            p = torch.tensor(pattern, device=self.device, dtype=torch.bfloat16)
            # self.weights += torch.outer(p, p) / len(data)
            self.weights += torch.outer(p, p)
        self.weights.diag().fill_(0)

    def add_noise(self, pattern, noise_level) -> torch.Tensor:
        noise = torch.random.choice([1, -1], pattern.shape, p=[1 - noise_level, noise_level], device=self.device)
        return pattern * noise

    def predict(self, pattern, iterations: int = 100):
        with torch.no_grad():
            if not isinstance(pattern, torch.Tensor):
                result = torch.tensor(pattern, device=self.device, dtype=torch.bfloat16)
            else:
                result = torch.tensor(pattern.cpu().numpy(), device=self.device, dtype=torch.bfloat16)
            for _ in tqdm(range(iterations)):
                result = torch.sign(self.weights @ result)
                # for i in range(self.size):
                #     result[i] = 1 if torch.dot(self.weights[i], result) > 0 else -1
            print(f"Result shape: {result.shape} result type: {result.dtype}")
            return result

    def flatten(self, image):
        if not isinstance(image, torch.Tensor):
            return torch.tensor(image.flatten(), device=self.device, dtype=torch.bfloat16)
        else:
            return image.flatten()


def add_noise(pattern, noise_level) -> torch.Tensor:
    # noise = torch.random.choice([1, -1], pattern.shape, p=[1 - noise_level, noise_level], device="cuda")
    noise = np.random.choice([1, -1], pattern.shape, p=[1 - noise_level, noise_level])
    return pattern * noise


def map_to_minus_one_to_one(image_array: torch.Tensor) -> torch.Tensor:
    # Normalize the array to the range [0, 1]
    normalized_array = image_array / 255.0

    # Map the normalized values to the range [-1, 1]
    mapped_array = 2 * normalized_array - 1

    return mapped_array


def map_to_zero_to_two_fifty_five(mapped_array):
    # Map the values from [-1, 1] to [0, 1]
    normalized_array = (mapped_array + 1) / 2.0

    # Scale the normalized values to the range [0, 255]
    image_array = (normalized_array * 255).astype(np.uint8)

    return image_array


def map_to_grayscale(mapped_array):
    # Map the values from [-1, 1] to [0, 1]
    normalized_array = (mapped_array + 1) / 2.0

    # Scale the normalized values to the range [0, 255]
    image_array = (normalized_array * 255).to(dtype=torch.uint8)

    return image_array


def load_dataset(load_prefix, resized_prefix, image_paths, max_size) -> tuple[list[np.ndarray], tuple[int, int]]:
    images = []
    shape = None
    for image_path in image_paths:
        image = dither_image(load_prefix + image_path)
        image = resize_image_array(image, max_size)
        shape = image.shape
        save_image_from_array(image, resized_prefix + image_path)
        images.append(image)
    return images, shape


def test_network(pattern, noise_level, hopfield_net, shape, output_noise, output_recovered):
    test_pattern = torch.tensor(hopfield_net.add_noise(pattern, noise_level))
    noise = torch.mean((test_pattern - pattern) ** 2)
    print(f"Noise level: {noise:.2f}")
    out_image = map_to_grayscale(test_pattern.reshape(shape))
    save_image_from_array(out_image.cpu().numpy(), output_noise)
    recovered_pattern = hopfield_net.predict(test_pattern)
    print(f"Recovered Pattern dtype: {recovered_pattern.dtype}")
    print(f"Pattern dtype: {pattern.dtype}")
    noise = torch.mean((recovered_pattern - torch.tensor(pattern, device="cuda", dtype=torch.bfloat16)) ** 2)
    print(f"Recovered Noise level: {noise:.2f}")
    # out_image = map_to_zero_to_two_fifty_five(recovered_pattern.reshape(image.shape))
    out_image = map_to_grayscale(recovered_pattern.reshape(shape))
    save_image_from_array(out_image.cpu().numpy(), output_recovered)


if __name__ == '__main__':
    # Configurar para CPU
    # max_size = 100
    # force_cpu = True
    # GPU 3090+
    max_size = 250
    force_cpu = False

    noise_level = 0.1

    print(f"Predicted RAM usage: {(max_size ** 2 ** 2 * 4 / 1024 / 1024):.0f} MB")

    print("Loading dataset ...")
    dataset, shape = load_dataset('images/resized/', 'images/hopfield/', ['Houssay.png', 'Leloir.png', 'Milstein.png'], max_size)
    print(f"Dataset: {len(dataset)} loaded.")
    # print(f"Shape: {shape}")
    # dataset, shape = load_dataset('images/resized/', 'images/hopfield/', ['Houssay.png'], max_size)

    # image = dither_image('images/resized/Houssay.png')
    # image = resize_image_array(image, max_size)
    # save_image_from_array(image, 'images/hopfield/Houssay_resized.jpg')

    # patterns = generate_pattern(size, num_patterns)
    # flattened = image.flatten()
    # print(f"Flattened shape: {shape}")
    # size = flattened.shape[0]
    # print(f"Size Squared: {size ** 2}")

    # print(f"Training Size:{flattened.shape} ...")
    hopfield_net = HopfieldNetworkTorch(shape[0]*shape[1], force_cpu=force_cpu)

    print("Mapping patterns ...")
    patterns = [map_to_minus_one_to_one(hopfield_net.flatten(image)) for image in dataset]
    print("Patterns mapped.")

    hopfield_net.train(patterns)

    print("Testing ...")
    test_network(patterns[0], noise_level, hopfield_net, shape, 'images/hopfield/Houssay_noise.jpg', 'images/hopfield/Houssay_recovered.jpg')
    test_network(patterns[1], noise_level*2, hopfield_net, shape, 'images/hopfield/Leloir_noise.jpg', 'images/hopfield/Leloir_recovered.jpg')
    test_network(patterns[2], noise_level*3, hopfield_net, shape, 'images/hopfield/Milstein_noise.jpg', 'images/hopfield/Milstein_recovered.jpg')
    # save_image_from_array(recovered_pattern.reshape(image.shape), 'images/Houssay_recovered.jpg')

    # print("Original pattern:  ", patterns[0])
    # print("Noisy pattern:     ", test_pattern)
    # print("Recovered pattern: ", recovered_pattern)
