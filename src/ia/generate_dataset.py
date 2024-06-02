from math import sqrt
from tqdm import tqdm
import drawsvg as draw
import os
from ia.image_loader import dither_image
import pickle

def generate_pngs():
    global x, y
    try:
        for filename in os.listdir("images/dataset-hopfield"):
            if not filename.startswith("."):
                os.remove(f"images/dataset-hopfield/{filename}")
    except FileNotFoundError:
        pass
    try:
        for filename in os.listdir("images/dataset-dithered"):
            if not filename.startswith("."):
                os.remove(f"images/dataset-dithered/{filename}")
    except FileNotFoundError:
        pass

    d = draw.Drawing(150, 150, origin='center')
    d.set_pixel_scale(2)  # Set number of pixels per geometry unit
    count = 0
    for x in tqdm(range(-20, 40, 20)):
        for y in range(0, 20, 20):
            for r in range(20, 30, 10):
                d.clear()
                c = draw.Rectangle(-75, -75, 150, 150, fill='#000000')
                # r.append_title("Our first rectangle")  # Add a tooltip
                d.append(c)
                d.append(draw.Circle(x, y, r,
                                     fill='black', stroke_width=4, stroke='white'))
                # d.set_render_size(400, 200)  # Alternative to set_pixel_scale
                # d.save_svg('example.svg')
                d.save_png(f'images/dataset-hopfield/circle_x{x}_y{y}_r{r}.png')
                image = dither_image(f"images/dataset-hopfield/circle_x{x}_y{y}_r{r}.png")
                with open(f"images/dataset-dithered/circle_x{x}_y{y}_r{r}.pkl", "wb") as f:
                    pickle.dump(image, f)
                count += 1
    print(f"Total images: {count}")
    hebb_size = count / 0.138
    minimum_size = sqrt(hebb_size)
    print(f"Hebbian size: {hebb_size:.1f} minimum side: {minimum_size:.1f}")


if __name__ == '__main__':
    generate_pngs()



