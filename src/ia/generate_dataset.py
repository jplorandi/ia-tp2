from math import sqrt
from tqdm import tqdm
import drawsvg as draw
import os

if __name__ == '__main__':
    for filename in os.listdir("images/dataset-hopfield"):
        os.remove(f"images/dataset-hopfield/{filename}")

    d = draw.Drawing(150, 150, origin='center')
    d.set_pixel_scale(2)  # Set number of pixels per geometry unit

    count = 0
    for x in tqdm(range(-48, 52, 4)):
        for y in range(-48, 52, 4):
            for r in range(10, 40, 10):
                d.clear()
                c = draw.Rectangle(-75, -75, 150, 150, fill='#000000')
                # r.append_title("Our first rectangle")  # Add a tooltip
                d.append(c)
                d.append(draw.Circle(x, y, r,
                                     fill='black', stroke_width=2, stroke='white'))
                # d.set_render_size(400, 200)  # Alternative to set_pixel_scale
                # d.save_svg('example.svg')
                d.save_png(f'images/dataset-hopfield/circle_x{x}_y{y}_r{r}.png')
                count += 1

    print(f"Total images: {count}")
    hebb_size = count / 0.138
    minimum_size = sqrt(hebb_size)
    print(f"Hebbian size: {hebb_size:.1f} minimum side: {minimum_size:.1f}")


    # d.fill(0,0,0)  # Set background color to white
    # Draw an irregular polygon
    # d.append(draw.Lines(-80, 45,
    #                      70, 49,
    #                      95, -49,
    #                     -90, -40,
    #                     close=False,
    #             fill='#eeee00',
    #             stroke='black'))
    #
    # # Draw a rectangle


    # Draw a circle
    d.append(draw.Circle(0, 0, 30,
            fill='black', stroke_width=2, stroke='white'))

    # # Draw an arbitrary path (a triangle in this case)
    # p = draw.Path(stroke_width=2, stroke='lime', fill='black', fill_opacity=0.2)
    # p.M(-10, -20)  # Start path at point (-10, -20)
    # p.C(30, 10, 30, -50, 70, -20)  # Draw a curve to (70, -20)
    # d.append(p)
    #
    # # Draw text
    # d.append(draw.Text('Basic text', 8, -10, -35, fill='blue'))  # 8pt text at (-10, -35)
    # d.append(draw.Text('Path text', 8, path=p, text_anchor='start', line_height=1))
    # d.append(draw.Text(['Multi-line', 'text'], 8, path=p, text_anchor='end', center=True))

    # Draw multiple circular arcs
    # d.append(draw.ArcLine(60, 20, 20, 60, 270,
    #         stroke='red', stroke_width=5, fill='red', fill_opacity=0.2))
    # d.append(draw.Arc(60, 20, 20, 90, -60, cw=True,
    #         stroke='green', stroke_width=3, fill='none'))
    # d.append(draw.Arc(60, 20, 20, -60, 90, cw=False,
    #         stroke='blue', stroke_width=1, fill='black', fill_opacity=0.3))

    # Draw arrows
    # arrow = draw.Marker(-0.1, -0.51, 0.9, 0.5, scale=4, orient='auto')
    # arrow.append(draw.Lines(-0.1, 0.5, -0.1, -0.5, 0.9, 0, fill='red', close=True))
    # p = draw.Path(stroke='red', stroke_width=2, fill='none',
    #         marker_end=arrow)  # Add an arrow to the end of a path
    # p.M(20, 40).L(20, 27).L(0, 20)  # Chain multiple path commands
    # d.append(p)
    # d.append(draw.Line(30, 20, 0, 10,
    #         stroke='red', stroke_width=2, fill='none',
    #         marker_end=arrow))  # Add an arrow to the end of a line



    # Display in Jupyter notebook
    #d.rasterize()  # Display as PNG
    #d  # Display as SVG
    # d.rasterize("circle.png")