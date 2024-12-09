import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import pnoise2


root = tk.Tk()
root.title("Glitch Effect with Adjustable Parameters")
root.geometry("1200x800")

original_image = None
processed_image = None

# Default Parameters
perlin_scale = 100
voronoi_points = 50
noise_octaves = 4
noise_persistence = 0.5
noise_lacunarity = 2.0
noise_seed = 42


def generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed):
    np.random.seed(seed)
    noise_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = pnoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )
    return (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))


def generate_voronoi_map(width, height, points, seed):
    np.random.seed(seed)
    point_coords = np.random.rand(points, 2) * [width, height]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    grid_coords = np.dstack((x_coords, y_coords)).reshape(-1, 2)
    distances = np.sum((grid_coords[:, np.newaxis] - point_coords) ** 2, axis=2)
    voronoi_map = np.argmin(distances, axis=1).reshape(height, width)
    return voronoi_map


def distort_image(image, displacement_x, displacement_y):
    height, width = image.shape[:2]
    distorted_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            distorted_image[y, x] = image[new_y, new_x]

    return distorted_image


def apply_glitch_effect(image, voronoi_map, perlin_map):
    displacement_x = (voronoi_map * 10 + perlin_map * 30).astype(np.int32)
    displacement_y = (perlin_map * 10 + voronoi_map * 30).astype(np.int32)
    return distort_image(image, displacement_x, displacement_y)


def apply_effect():
    if original_image is None:
        return

    scale = perlin_scale_slider.get()
    points = voronoi_points_slider.get()
    octaves = noise_octaves_slider.get()
    persistence = noise_persistence_slider.get() / 100
    lacunarity = noise_lacunarity_slider.get() / 10
    seed = noise_seed_slider.get()

    height, width = original_image.shape[:2]
    perlin_map = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed)
    voronoi_map = generate_voronoi_map(width, height, points, seed)

    processed = apply_glitch_effect(original_image, voronoi_map, perlin_map)
    update_display_image(processed)


def update_display_image(image):
    global processed_image
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(processed_image))
    image_label.config(image=img_tk)
    image_label.image = img_tk


def load_image():
    global original_image
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not filepath:
        return

    loaded_image = cv2.imread(filepath)
    height, width = loaded_image.shape[:2]

    if height > 400 or width > 400:
        scale_factor = min(400 / width, 400 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        original_image = cv2.resize(loaded_image, (new_width, new_height))
    else:
        original_image = loaded_image

    apply_effect()


main_frame = tk.Frame(root, bg="#2e2e2e")
main_frame.pack(fill="both", expand=True)

# Left frame for controls
left_frame = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame.pack(side="left", fill="y", expand=True)

# Load Image Button
button_frame = tk.Frame(left_frame, bg="#2e2e2e")
button_frame.pack(fill="x", pady=10)
load_button = tk.Button(button_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(padx=10, pady=5)

# Perlin Noise Scale Slider
perlin_scale_slider = tk.Scale(left_frame, from_=10, to=200, label="Perlin Scale", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
perlin_scale_slider.set(perlin_scale)
perlin_scale_slider.pack(fill="x", pady=5)

# Voronoi Points Slider
voronoi_points_slider = tk.Scale(left_frame, from_=10, to=200, label="Voronoi Points", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
voronoi_points_slider.set(voronoi_points)
voronoi_points_slider.pack(fill="x", pady=5)

# Noise Octaves Slider
noise_octaves_slider = tk.Scale(left_frame, from_=1, to=10, label="Noise Octaves", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
noise_octaves_slider.set(noise_octaves)
noise_octaves_slider.pack(fill="x", pady=5)

# Noise Persistence Slider
noise_persistence_slider = tk.Scale(left_frame, from_=1, to=100, label="Noise Persistence (%)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
noise_persistence_slider.set(int(noise_persistence * 100))
noise_persistence_slider.pack(fill="x", pady=5)

# Noise Lacunarity Slider
noise_lacunarity_slider = tk.Scale(left_frame, from_=10, to=50, label="Noise Lacunarity (/10)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
noise_lacunarity_slider.set(int(noise_lacunarity * 10))
noise_lacunarity_slider.pack(fill="x", pady=5)

# Noise Seed Slider
noise_seed_slider = tk.Scale(left_frame, from_=0, to=100, label="Noise Seed", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_effect())
noise_seed_slider.set(noise_seed)
noise_seed_slider.pack(fill="x", pady=5)

# Right frame for image display
right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
