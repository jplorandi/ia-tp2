import numpy as np
from hflayers.auxiliary.data import BitPatternSet
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from tqdm import tqdm
from image_loader import dither_image, save_image_from_array, resize_image_array
import torch
import os
from hflayers import Hopfield
from dataclasses import dataclass
import pickle
import pandas as pd
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential, Sigmoid
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from typing import Optional, Tuple

@dataclass(frozen=True, eq=True, order=True)
class Circle:
    x: int
    y: int
    r: int
    tensor: torch.Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, dataset: dict, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label
        return self.dataset[idx]


def load_dataset_simple(size, device):
    dataset = {}
    for x in tqdm(range(-40, 50, 10)):
        for y in range(-40, 50, 10):
            for r in range(10, 20, 10):
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


def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader
                ) -> Tuple[float, float, float]:
    """
    Execute one training epoch.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss, training error as well as accuracy
    """
    network.train()
    losses, errors, accuracies = [], [], []
    for data, target in data_loader:
        data, target = data.to(device=device), target[0].to(device=device)

        # Process data by Hopfield-based network.
        loss = network.calculate_objective(data, target)[0]

        # Update network parameters.
        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        error, prediction = network.calculate_classification_error(data, target)
        accuracy = (prediction == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        errors.append(error)
        losses.append(loss.detach().item())

    # Report progress of training procedure.
    return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)


def eval_iter(network: Module,
              data_loader: DataLoader
              ) -> Tuple[float, float, float]:
    """
    Evaluate the current model.

    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss, validation error as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, errors, accuracies = [], [], []
        for data, target in data_loader:
            data, target = data.to(device=device), target[0].to(device=device)

            # Process data by Hopfield-based network.
            loss = network.calculate_objective(data, target)[0]

            # Compute performance measures of current model.
            error, prediction = network.calculate_classification_error(data, target)
            accuracy = (prediction == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            errors.append(error)
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return sum(losses) / len(losses), sum(errors) / len(errors), sum(accuracies) / len(accuracies)


def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.

    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, errors, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        losses[r'train'].append(performance[0])
        errors[r'train'].append(performance[1])
        accuracies[r'train'].append(performance[2])

        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval)
        losses[r'eval'].append(performance[0])
        errors[r'eval'].append(performance[1])
        accuracies[r'eval'].append(performance[2])

    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(errors), pd.DataFrame(accuracies)

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
    max_size = 64
    force_cpu = False

    noise_level = 0.45

    print(f"Predicted RAM usage: {(max_size ** 2 ** 2 * 4 / 1024 / 1024):.0f} MB")

    print("Loading dataset ...")
    for file in os.listdir('images/hopfield'):
        os.remove(f"images/hopfield/{file}")
    # dataset, shape = load_dataset('images/resized/', 'images/hopfield/', ['Houssay.png', 'Leloir.png', 'Milstein.png'], max_size)
    dataset = load_dataset_simple(max_size, torch.device("cuda"))
    print(f"Dataset: {len(dataset)} loaded.")
    cd = CustomImageDataset(dataset)

    print(f"Dataset: {len(dataset)} loaded.")

    # hopfield_net = HopfieldNetworkTorch(shape[0]*shape[1], force_cpu=force_cpu, dtype=torch.float32)
    hop = Hopfield(max_size*max_size)

    sampler_train = SubsetRandomSampler(list(range(256, 2048 - 256)))
    data_loader_train = DataLoader(dataset=dataset, batch_size=1, sampler=sampler_train)

    # Create data loader of validation set.
    sampler_eval = SubsetRandomSampler(list(range(256)) + list(range(2048 - 256, 2048)))
    data_loader_eval = DataLoader(dataset=dataset, batch_size=1, sampler=sampler_eval)

    # print("Mapping patterns ...")
    # patterns = [map_to_minus_one_to_one(image.flatten()) for image in dataset]
    # print("Patterns mapped.")

    # hopfield_net.train_pi(patterns)
    res = train_epoch(hop, AdamW(hop.parameters()), data_loader_train)
    # res = hop.train()
    print(f"Training done! {res}")

    prj = hop.forward(dataset)
    print(f"Projection: {prj}")


    print("Testing ...")
    # test_network(patterns[0], noise_level, hopfield_net, shape, 'images/hopfield/Houssay_noise.png', 'images/hopfield/Houssay_recovered.png')
    # test_network(patterns[1], noise_level, hopfield_net, shape, 'images/hopfield/Leloir_noise.png', 'images/hopfield/Leloir_recovered.png')
    # test_network(patterns[2], noise_level, hopfield_net, shape, 'images/hopfield/Milstein_noise.png', 'images/hopfield/Milstein_recovered.png')
