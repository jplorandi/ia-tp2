import os
import random
import pickle
from tqdm import tqdm
from dataclasses import dataclass
import torch
from ia.image_loader import load_image, dither_image, save_image_from_array, resize_image_array
from ia.hopfield import map_to_minus_one_to_one, HopfieldNetworkTorch, map_to_grayscale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True, eq=True, order=True)
class Circle:
    x: int
    y: int
    r: int
    tensor: torch.Tensor


def load_dataset(size):
    dataset = {}
    try:
        for filename in os.listdir("images/dataset-mirror"):
            if not filename.startswith("."):
                os.remove(f"images/dataset-mirror/{filename}")
    except FileNotFoundError:
        pass

    for x in tqdm(range(-20, 40, 20)):
        for y in range(0, 20, 20):
            for r in range(20, 30, 10):
                # image = load_image(f"images/dataset-hopfield/circle_x{x}_y{y}_r{r}.png")
                with open(f"images/dataset-dithered/circle_x{x}_y{y}_r{r}.pkl", "rb") as f:
                    image = pickle.load(f)
                image = resize_image_array(image, size)
                save_image_from_array(image, f"images/dataset-mirror/mirror-circle_x{x}_y{y}_r{r}.png")
                tensor = torch.tensor(map_to_minus_one_to_one(image.flatten()),
                                      device=device, dtype=torch.bfloat16)
                # pixel_count = torch.sum(tensor == 1)
                # pixel_black_count = torch.sum(tensor == -1)
                # print(f"Circle: {x},{y},{r} Pixel count: {pixel_count} Black count: {pixel_black_count}")
                save_image_from_array(map_to_grayscale(tensor.reshape(size, size)).cpu().numpy(), f"images/dataset-mirror/mirror2-circle_x{x}_y{y}_r{r}.png")
                circle = Circle(x, y, r, tensor)
                dataset[f"{x}_{y}_{r}"] = circle
    return dataset

def random_circle():
    x = 0
    y = 0
    r = 0
    while x == 0 and y == 0 and r == 0:
        x = random.randint(0,2) * 20 - 20
        y = 0
        r = 20
    return x, y, r


if __name__ == '__main__':
    size = 150
    dataset = load_dataset(size)
    print(f"Total images: {len(dataset)}")
    hebb_size = len(dataset) / 0.138
    minimum_size = hebb_size ** 0.5
    print(f"Hebbian size: {hebb_size:.1f} minimum side: {minimum_size:.1f}")
    # print(f"Dataset: {dataset}")

    hopfield_net = HopfieldNetworkTorch(size ** 2)

    patterns = [circle.tensor for circle in dataset.values()]
    patterns = patterns[:3]
    # print(len(patterns))
    hopfield_net.train_pi(patterns)
    # hopfield_net.train(patterns)
    print("Training done!")

    for file in os.listdir("images/predictions"):
        os.remove(f"images/predictions/{file}")

    for _ in range(5):
        x,y,r = random_circle()
        circle = dataset[f"{x}_{y}_{r}"]
        noisy_pattern = hopfield_net.add_noise(circle.tensor, 0.01)
        save_image_from_array(map_to_grayscale(noisy_pattern.reshape(size, size)).cpu().numpy(),
                              f"images/predictions/noisy_circle_x{x}_y{y}_r{r}.png")
        save_image_from_array(map_to_grayscale(circle.tensor.reshape(size, size)).cpu().numpy(),
                              f"images/predictions/orig_circle_x{x}_y{y}_r{r}.png")
        recovered_pattern = hopfield_net.predict(noisy_pattern, 50)
        save_image_from_array(map_to_grayscale(recovered_pattern.reshape(size, size)).cpu().numpy(),
                              f"images/predictions/circle_x{x}_y{y}_r{r}.png")
        mse = torch.mean((recovered_pattern - circle.tensor) ** 2)
        pixel_count = torch.sum(recovered_pattern > 0)
        black_count = torch.sum(recovered_pattern <= 0)
        print(f"Circle: {x},{y},{r} MSE: {mse:.2f} pixel count: {pixel_count} black count: {black_count}")
        # for circle2 in dataset.values():
        #     mse2 = torch.mean((recovered_pattern - circle2.tensor) ** 2)
        #     # print(f"Circle: {circle2.x},{circle2.y},{circle2.r} MSE: {mse:.2f}")
        #     if mse2 < mse and circle != circle2:
        #         print(f"Recovered: {circle2.x},{circle2.y},{circle2.r} MSE: {mse2:.2f} < {mse:.2f} Circle: {x},{y},{r}")
