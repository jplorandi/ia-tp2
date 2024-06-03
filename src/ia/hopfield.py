import numpy as np
from tqdm import tqdm
from image_loader import dither_image, save_image_from_array, resize_image_array
import torch
import os


class HopfieldNetworkTorch:
    def __init__(self, size, force_cpu=False, dtype=torch.float32):
        self.size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() and force_cpu is False else "cpu")
        self.dtype = dtype
        self.weights = torch.zeros((size, size), device=self.device, dtype=self.dtype)

    def train(self, data):
        for pattern in tqdm(data):
            p = pattern
            count = torch.sum(p == 1) + torch.sum(p == -1)
            if (count != self.size):
                print(f"Pattern count mismatch: {count}")

            if not isinstance(pattern, torch.Tensor):
                # p = torch.tensor(pattern, device=self.device, dtype=torch.bfloat16)
                raise ValueError("Pattern must be a tensor")

            # self.weights += torch.outer(p, p) / len(data)
            # w = p * p.t()
            self.weights += torch.outer(p, p)

        # count = torch.sum(self.weights == 1) + torch.sum(self.weights == -1)
        # if (count != self.size):
        #     print(f"W Pattern count mismatch: {count}")
        self.weights /= len(data)
        self.weights.diag().fill_(0)

    # def train_pi(self, data):
    #     A = torch.zeros_like(self.weights, device=self.device, dtype=torch.float32)
    #
    #     for i, pattern in tqdm(enumerate(data)):
    #         A[i] = pattern
    #
    #     A_p = torch.linalg.pinv(A)
    #
    #     self.weights += torch.linalg.lstsq(A, A_p).solution
    #     # self.weights.diag().fill_(0)

    def train_pi(self, patterns):
        # num_patterns = len(patterns)
        print(f"Patterns: {len(patterns)} size: {self.size}")
        extra = [torch.zeros_like(patterns[0], device=self.device, dtype=self.dtype) for _ in
                 range(self.size - len(patterns))]
        print(f"Extra: {len(extra) + len(patterns)}")
        A = torch.stack(patterns + extra, dim=0)
        print(f"A shape: {A.shape}")
        self.weights = torch.matmul(A.t(), A) / self.size
        self.weights.diag().fill_(0)  # Set diagonal elements to zero

    def add_noise(self, pattern, noise_level) -> torch.Tensor:
        values = torch.tensor([1, -1], device=self.device, dtype=self.dtype)
        probabilities = torch.tensor([1 - noise_level, noise_level], device=self.device, dtype=self.dtype)

        # Generate random choices
        random_choices = torch.multinomial(probabilities, pattern.numel(), replacement=True).reshape(pattern.shape)

        # Map the indices to the actual values
        random_array = values[random_choices]

        # print(random_array)

        # noise = (torch.randint(0, 2, pattern.shape, device=self.device, dtype=torch.bfloat16) * 2) - 1
        return pattern * random_array

    def predict(self, pattern, iterations: int = 100):
        mean_weights = torch.mean(self.weights)
        sum_weights = torch.sum(self.weights)
        print(f"Mean weights: {mean_weights:.2f} Sum weights: {sum_weights:.2f}")
        one_count = torch.sum(self.weights >= 1)
        minus_one_count = torch.sum(self.weights <= -1)
        # zero_count = torch.sum(-1 < self.weights < 1)
        print(f"One count: {one_count} Minus one count: {minus_one_count}")

        # with torch.no_grad():
        pixel_count = torch.sum(pattern > 0)
        print(f"Pixel count: {pixel_count}")
        result = pattern.clone()
        if not isinstance(pattern, torch.Tensor):
            raise ValueError("Pattern must be a tensor")
        # if not isinstance(pattern, torch.Tensor):
        #     result = torch.tensor(pattern, device=self.device, dtype=torch.bfloat16)
        # else:
        #     result = pattern.clone()
        for _ in tqdm(range(iterations)):
            output_pattern = torch.sign(self.weights @ result)
            if torch.equal(output_pattern, result):
                return output_pattern
            result = output_pattern

            # result = self.weights @ result
            # result[result >= 0] = 1
            # result[result < 0] = -1

            # pixel_count = torch.sum(result > 0)
            # black_count = torch.sum(result < 0)
            # nil_count = torch.sum(result == 0)
            # print(f"iPixel {_} count: {pixel_count} Black count: {black_count} Nil count: {nil_count}")

            # for i in range(self.size):
            #     result[i] = 1 if torch.dot(self.weights[i], result) > 0 else -1
        # print(f"Result shape: {result.shape} result type: {result.dtype}")
        return result

    def flatten(self, image):
        if not isinstance(image, torch.Tensor):
            return torch.tensor(image.flatten(), device=self.device, dtype=self.dtype)
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
    test_pattern = hopfield_net.add_noise(pattern, noise_level)
    noise = torch.mean((test_pattern - pattern) ** 2)
    print(f"Noise level: {noise:.2f}")
    out_image = map_to_grayscale(test_pattern.reshape(shape))
    save_image_from_array(out_image.cpu().numpy(), output_noise)
    recovered_pattern = hopfield_net.predict(test_pattern)
    # print(f"Recovered Pattern dtype: {recovered_pattern.dtype}")
    # print(f"Pattern dtype: {pattern.dtype}")
    noise = torch.mean((recovered_pattern - pattern) ** 2)
    print(f"Recovered Noise level: {noise:.2f}")
    # out_image = map_to_zero_to_two_fifty_five(recovered_pattern.reshape(image.shape))
    out_image = map_to_grayscale(recovered_pattern.reshape(shape))
    save_image_from_array(out_image.cpu().numpy(), output_recovered)


if __name__ == '__main__':
    # Configurar para CPU
    # max_size = 100
    # force_cpu = True
    # GPU 3090+
    max_size = 150
    force_cpu = False

    noise_level = 0.45

    print(f"Predicted RAM usage: {(max_size ** 2 ** 2 * 4 / 1024 / 1024):.0f} MB")

    print("Loading dataset ...")
    for file in os.listdir('images/hopfield'):
        os.remove(f"images/hopfield/{file}")
    dataset, shape = load_dataset('images/resized/', 'images/hopfield/', ['Houssay.png', 'Leloir.png', 'Milstein.png'],
                                  max_size)
    print(f"Dataset: {len(dataset)} loaded.")

    hopfield_net = HopfieldNetworkTorch(shape[0] * shape[1], force_cpu=force_cpu, dtype=torch.float32)

    print("Mapping patterns ...")
    patterns = [map_to_minus_one_to_one(hopfield_net.flatten(image)) for image in dataset]
    print("Patterns mapped.")

    hopfield_net.train_pi(patterns)

    print("Testing ...")
    test_network(patterns[0], noise_level, hopfield_net, shape, 'images/hopfield/Houssay_noise.png',
                 'images/hopfield/Houssay_recovered.png')
    test_network(patterns[1], noise_level, hopfield_net, shape, 'images/hopfield/Leloir_noise.png',
                 'images/hopfield/Leloir_recovered.png')
    test_network(patterns[2], noise_level, hopfield_net, shape, 'images/hopfield/Milstein_noise.png',
                 'images/hopfield/Milstein_recovered.png')
