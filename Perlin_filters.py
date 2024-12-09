import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import pnoise2

root = tk.Tk()
root.title("Perlin Noise Texture Simulations")
root.geometry("1200x800")

original_image = None
processed_image = None

# Default Perlin parameters
perlin_scale = 100
perlin_octaves = 4
perlin_persistence = 1
perlin_lacunarity = 2.0

# Default texture-specific parameters
marble_frequency = 10
wave_strength = 30

# Active textures
active_textures = {
    "Marble": False,
    "Waves": False,
}


def generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed=0):
    noise_map = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            noise_map[y, x] = pnoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )

    # Normalize the noise to range [0, 1]
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return noise_map


def distort_image(image, noise_map, strength=20):
    """
    Distorts the image using the noise map as a displacement field.
    """
    height, width = image.shape[:2]
    displacement_x = (noise_map * strength).astype(np.float32)
    displacement_y = (noise_map * strength).astype(np.float32)

    # Create remapping grids
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (map_x + displacement_x).astype(np.float32)
    map_y = (map_y + displacement_y).astype(np.float32)

    # Apply distortion
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_marble_texture(image, noise_map):
    marble_map = np.sin(marble_frequency_slider.get() * noise_map + noise_map * 5)  # Adjust frequency and distortion
    marble_map = (marble_map - np.min(marble_map)) / (np.max(marble_map) - np.min(marble_map))
    return distort_image(image, marble_map, strength=15)


def apply_wave_texture(image, noise_map):
    wave_map = np.sin(10 * noise_map) + noise_map
    wave_map = (wave_map - np.min(wave_map)) / (np.max(wave_map) - np.min(wave_map))
    return distort_image(image, wave_map, strength=wave_strength_slider.get())


def apply_selected_textures():
    if original_image is None:
        return

    # Retrieve Perlin parameters from sliders
    scale = perlin_scale_slider.get()
    octaves = perlin_octaves_slider.get()
    persistence = perlin_persistence_slider.get() / 100
    lacunarity = perlin_lacunarity_slider.get() / 10

    height, width = original_image.shape[:2]
    noise_map = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity)

    processed = original_image.copy()

    if active_textures["Marble"]:
        processed = apply_marble_texture(processed, noise_map)
    if active_textures["Waves"]:
        processed = apply_wave_texture(processed, noise_map)

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

    apply_selected_textures()


def toggle_texture(texture_name):
    active_textures[texture_name] = not active_textures[texture_name]
    apply_selected_textures()


main_frame = tk.Frame(root, bg="#2e2e2e")
main_frame.pack(fill="both", expand=True)

# Left frame for controls
left_frame = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame.pack(side="left", fill="y", expand=True)

# Perlin noise sliders
perlin_scale_slider = tk.Scale(left_frame, from_=10, to=200, label="Perlin Scale", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
perlin_scale_slider.set(perlin_scale)
perlin_scale_slider.pack(fill="x", pady=5)

perlin_octaves_slider = tk.Scale(left_frame, from_=1, to=10, label="Perlin Octaves", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
perlin_octaves_slider.set(perlin_octaves)
perlin_octaves_slider.pack(fill="x", pady=5)

perlin_persistence_slider = tk.Scale(left_frame, from_=10, to=100, label="Perlin Persistence (%)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
perlin_persistence_slider.set(int(perlin_persistence * 100))
perlin_persistence_slider.pack(fill="x", pady=5)

perlin_lacunarity_slider = tk.Scale(left_frame, from_=10, to=50, label="Perlin Lacunarity (/10)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
perlin_lacunarity_slider.set(int(perlin_lacunarity * 10))
perlin_lacunarity_slider.pack(fill="x", pady=5)

# Texture-specific sliders
marble_frequency_slider = tk.Scale(left_frame, from_=1, to=20, label="Marble Frequency", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
marble_frequency_slider.set(marble_frequency)
marble_frequency_slider.pack(fill="x", pady=5)

wave_strength_slider = tk.Scale(left_frame, from_=1, to=50, label="Wave Strength", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_textures())
wave_strength_slider.set(wave_strength)
wave_strength_slider.pack(fill="x", pady=5)

# Texture checkboxes
for texture_name in active_textures:
    checkbox = tk.Checkbutton(left_frame, text=texture_name, bg="#2e2e2e", fg="white",selectcolor="#2e2e2e", command=lambda name=texture_name: toggle_texture(name))
    checkbox.pack(anchor="w", padx=10)

# Button to load image
load_button = tk.Button(left_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(fill="x", pady=10)

# Right frame for image display
right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
