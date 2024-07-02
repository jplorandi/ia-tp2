import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def hough_circle_transform(edges, r_min, r_max):
    rows, cols = edges.shape
    max_radius = r_max
    accumulator = torch.zeros((rows, cols, max_radius), dtype=torch.int32, device=device)

    edge_points = torch.nonzero(torch.tensor(edges))
    thetas = torch.arange(0, 360).float().to(device)
    cos_thetas = torch.cos(torch.deg2rad(thetas)).to(device)
    sin_thetas = torch.sin(torch.deg2rad(thetas)).to(device)

    for x, y in tqdm(edge_points):
        x, y = x.item(), y.item()
        for r in range(r_min, r_max):
            b_values = y - (r * sin_thetas).int()
            a_values = x - (r * cos_thetas).int()
            valid_indices = (a_values >= 0) & (a_values < rows) & (b_values >= 0) & (b_values < cols)
            valid_a_values = a_values[valid_indices]
            valid_b_values = b_values[valid_indices]
            for a, b in zip(valid_a_values, valid_b_values):
                accumulator[a, b, r] += 1

    return accumulator


def detect_circles(accumulator, threshold):
    detected_circles = torch.nonzero(accumulator > threshold)
    return detected_circles


def draw_circles(image, circles):
    for circle in circles:
        a, b, r = circle
        cv2.circle(image, (b.item(), a.item()), r.item(), (0, 255, 0), 2)
        cv2.circle(image, (b.item(), a.item()), 2, (0, 0, 255), 3)
    return image


def main(image_path, r_min, r_max, threshold):
    image = cv2.imread(image_path)
    print(f"Image shape: {image.shape}")
    edges = edge_detection(image)
    print(f"Edges shape: {edges.shape}")
    accumulator = hough_circle_transform(edges, r_min, r_max)
    print(f"Accumulator shape: {accumulator.shape}")
    circles = detect_circles(accumulator, threshold)
    print(f"Detected circles: {circles.shape}")

    result_image = draw_circles(image, circles)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig('images/hough/HoughCirclesDetected.jpg')


if __name__ == '__main__':
    # Example usage
    main('images/hough/HoughCircles.jpg', r_min=10, r_max=50, threshold=150)
