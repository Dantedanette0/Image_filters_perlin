import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import snoise2

root = tk.Tk()
root.title("Simplex Noise Filters")
root.geometry("1200x800")

original_image = None
processed_image = None

# Default Parameters for Simplex noise
simplex_scale = 100
simplex_octaves = 4
simplex_persistence = 0.5
simplex_lacunarity = 2.0
color_intensity = 100  # Adjusts the strength of the color distortions
warp_intensity = 50  # Intensity of dynamic warping

# Active filters
active_filters = {
    "Dynamic Warping": False,
    "Abstract Color Effect": False,
    "Invert Colors": False,
    "Layered and Glitch Effects": False,
}


def generate_simplex_noise_map(width, height, scale, octaves, persistence, lacunarity, seed=0):
    noise_map = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            noise_map[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )

    # Normalize the noise to range [0, 1]
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return noise_map


def apply_dynamic_warping(image, noise_map, intensity):
    height, width = image.shape[:2]
    displacement_x = (noise_map * intensity).astype(np.int32)
    displacement_y = (noise_map * intensity).astype(np.int32)

    warped_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            warped_image[y, x] = image[new_y, new_x]

    return warped_image

def generate_simplex_color_map(width, height, scale, octaves, persistence, lacunarity, seed, intensity):
    color_map_r = np.zeros((height, width), dtype=np.float32)
    color_map_g = np.zeros((height, width), dtype=np.float32)
    color_map_b = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Generate noise for each color channel
            color_map_r[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )
            color_map_g[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 1
            )
            color_map_b[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 2
            )

    # Normalize maps to the range [0, 1]
    color_map_r = (color_map_r - np.min(color_map_r)) / (np.max(color_map_r) - np.min(color_map_r))
    color_map_g = (color_map_g - np.min(color_map_g)) / (np.max(color_map_g) - np.min(color_map_g))
    color_map_b = (color_map_b - np.min(color_map_b)) / (np.max(color_map_b) - np.min(color_map_b))

    # Scale maps to intensity
    color_map_r = (color_map_r * intensity).astype(np.int32)
    color_map_g = (color_map_g * intensity).astype(np.int32)
    color_map_b = (color_map_b * intensity).astype(np.int32)

    return color_map_r, color_map_g, color_map_b


def apply_abstract_color_effect(image, scale, octaves, persistence, lacunarity, seed, intensity):
    height, width = image.shape[:2]
    color_map_r, color_map_g, color_map_b = generate_simplex_color_map(
        width, height, scale, octaves, persistence, lacunarity, seed, intensity
    )
    b, g, r = cv2.split(image)
    r = np.clip(r + color_map_r, 0, 255).astype(np.uint8)
    g = np.clip(g + color_map_g, 0, 255).astype(np.uint8)
    b = np.clip(b + color_map_b, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def apply_Invert_effects(image, noise_map):
 
    b, g, r = cv2.split(image)  # Split the image into Blue, Green, Red channels
    b = np.clip(b + (noise_map * 50).astype(np.uint8), 0, 255)  # Modify Blue channel
    g = np.clip(g + (noise_map * 50).astype(np.uint8), 0, 255)  # Modify Green channel
    r = np.clip(r + (noise_map * 50).astype(np.uint8), 0, 255)  # Modify Red channel
    return cv2.merge((b, g, r))  # Merge the channels back


def apply_layered_and_glitch_effects(image, noise_map):
    layered_map = (noise_map + noise_map * 0.5) % 1
    layered_map = (layered_map * 255).astype(np.uint8)

    layered_map_rgb = cv2.cvtColor(layered_map, cv2.COLOR_GRAY2BGR)

    orange_tint = np.zeros_like(layered_map_rgb)
    orange_tint[:, :, 1] = 100
    orange_tint[:, :, 2] = 150

    layered_map_rgb = cv2.addWeighted(layered_map_rgb, 0.7, orange_tint, 0.3, 0)

    mask = (layered_map > 128).astype(np.uint8)
    glitch_effect = cv2.bitwise_and(image, layered_map_rgb, mask=mask)

    return glitch_effect


def apply_selected_filters():
    if original_image is None:
        return

    scale = simplex_scale_slider.get()
    octaves = simplex_octaves_slider.get()
    persistence = simplex_persistence_slider.get() / 100
    lacunarity = simplex_lacunarity_slider.get() / 10
    color_intensity = color_intensity_slider.get()
    warp = warp_intensity_slider.get()

    height, width = original_image.shape[:2]
    noise_map = generate_simplex_noise_map(width, height, scale, octaves, persistence, lacunarity)

    processed = original_image.copy()
    if active_filters["Dynamic Warping"]:
        processed = apply_dynamic_warping(processed, noise_map, warp)
    if active_filters["Abstract Color Effect"]:
        processed = apply_abstract_color_effect(processed, scale, octaves, persistence, lacunarity, 0,color_intensity)
    if active_filters["Invert Colors"]:
        processed = apply_Invert_effects(processed, noise_map)
    if active_filters["Layered and Glitch Effects"]:
        processed = apply_layered_and_glitch_effects(processed, noise_map)

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

    apply_selected_filters()


def toggle_filter(filter_name):
    active_filters[filter_name] = not active_filters[filter_name]
    apply_selected_filters()


main_frame = tk.Frame(root, bg="#2e2e2e")
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame.pack(side="left", fill="y", expand=True)

simplex_scale_slider = tk.Scale(left_frame, from_=1, to=200, label="Scale", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
simplex_scale_slider.set(simplex_scale)
simplex_scale_slider.pack(fill="x", pady=5)

simplex_octaves_slider = tk.Scale(left_frame, from_=1, to=10, label="Octaves", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
simplex_octaves_slider.set(simplex_octaves)
simplex_octaves_slider.pack(fill="x", pady=5)

simplex_persistence_slider = tk.Scale(left_frame, from_=1, to=100, label="Persistence (%)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
simplex_persistence_slider.set(int(simplex_persistence * 100))
simplex_persistence_slider.pack(fill="x", pady=5)

simplex_lacunarity_slider = tk.Scale(left_frame, from_=1, to=30, label="Lacunarity (/10)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
simplex_lacunarity_slider.set(int(simplex_lacunarity * 10))
simplex_lacunarity_slider.pack(fill="x", pady=5)

color_intensity_slider = tk.Scale(left_frame, from_=1, to=100, label="Color Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
color_intensity_slider.set(color_intensity)
color_intensity_slider.pack(fill="x", pady=5)

warp_intensity_slider = tk.Scale(left_frame, from_=1, to=100, label="Warp Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
warp_intensity_slider.set(warp_intensity)
warp_intensity_slider.pack(fill="x", pady=5)

for filter_name in active_filters:
    checkbox = tk.Checkbutton(left_frame, text=filter_name, bg="#2e2e2e", fg="white", selectcolor="#2e2e2e", command=lambda name=filter_name: toggle_filter(name))
    checkbox.pack(anchor="w", padx=10)

load_button = tk.Button(left_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(fill="x", pady=10)

right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
