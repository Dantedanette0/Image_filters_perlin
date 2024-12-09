import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import pnoise2

root = tk.Tk()
root.title("Perlin Noise Effects with Hallucination Effect")
root.geometry("1200x825")

original_image = None
processed_image = None

# Default Parameters
perlin_scale = 100
perlin_octaves = 4
perlin_persistence = 0.5
perlin_lacunarity = 2.0
lightning_intensity = 5.0
displacement_intensity = 0.1
marble_frequency = 10
wave_strength = 30
cloud_opacity = 0.5
wave_amplitude = 15
wave_frequency = 30
shift_r = 5
shift_g = -5
shift_b = 10

# Active Effects
active_effects = {
    "Lightning": False,
    "Pixel Swapping": False,
    "Marble": False,
    "Waves": False,
    "Clouds": False,
    "Hallucination": False,
}

# Perlin Noise Map Generation
def generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed=0):
    noise_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = pnoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )
    return (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

# Lightning Effect
def apply_lightning_effect(image, scale, octaves, persistence, lacunarity, intensity):
    height, width = image.shape[:2]
    lightning_map = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity)
    lightning_map = np.power(lightning_map, 6)
    lightning_map_rgb = (lightning_map * 255).astype(np.uint8)
    lightning_map_rgb = cv2.cvtColor(lightning_map_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1.0, lightning_map_rgb, intensity, 0)

# Pixel Swapping
def apply_pixel_swapping(image, scale, octaves, persistence, lacunarity, seed, intensity):
    height, width = image.shape[:2]
    displacement_x = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed) * width * intensity
    displacement_y = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity, seed + 1) * height * intensity
    displacement_x = displacement_x.astype(np.int32)
    displacement_y = displacement_y.astype(np.int32)
    swapped_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            swapped_image[y, x] = image[new_y, new_x]
    return swapped_image

# Marble Texture
def apply_marble_texture(image, noise_map, frequency):
    marble_map = np.sin(frequency * noise_map + noise_map * 5)
    marble_map = (marble_map - np.min(marble_map)) / (np.max(marble_map) - np.min(marble_map))
    return distort_image(image, marble_map, strength=15)

# Wave Texture
def apply_wave_texture(image, noise_map, strength):
    wave_map = np.sin(10 * noise_map) + noise_map
    wave_map = (wave_map - np.min(wave_map)) / (np.max(wave_map) - np.min(wave_map))
    return distort_image(image, wave_map, strength=strength)

# Cloud Effect
def apply_cloud_effect(image, scale, octaves, persistence, lacunarity, opacity):
    height, width = image.shape[:2]
    cloud_map = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity)
    cloud_map_rgb = (cloud_map * 255).astype(np.uint8)
    cloud_map_rgb = cv2.cvtColor(cloud_map_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - opacity, cloud_map_rgb, opacity, 0)

def apply_wave_distortion(image, amplitude, frequency):

    height, width = image.shape[:2]
    distorted_image = np.zeros_like(image)
    for y in range(height):
        offset = int(amplitude * np.sin(2 * np.pi * y / frequency))
        distorted_image[y, :] = np.roll(image[y, :], offset, axis=0)
    return distorted_image


def apply_color_shift(image, shift_r, shift_g, shift_b):

    b, g, r = cv2.split(image)
    r = np.roll(r, shift_r, axis=1)
    g = np.roll(g, shift_g, axis=0)
    b = np.roll(b, shift_b, axis=1)
    return cv2.merge((b, g, r))

def generate_perlin_noise_map_for_hallucination(width, height, scale=50):

    noise_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = pnoise2(x / scale, y / scale)
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return noise_map

def apply_hallucination_effect(image, wave_amplitude, wave_frequency, perlin_scale, shift_r, shift_g, shift_b):

    # Apply wave distortion
    distorted_image = apply_wave_distortion(image, amplitude=wave_amplitude, frequency=wave_frequency)

    # Apply color shift
    color_shifted_image = apply_color_shift(distorted_image, shift_r=shift_r, shift_g=shift_g, shift_b=shift_b)

    # Generate Perlin noise overlay
    height, width = image.shape[:2]
    perlin_map = generate_perlin_noise_map_for_hallucination(width, height,scale=perlin_scale)
    perlin_overlay = (perlin_map * 255).astype(np.uint8)
    perlin_overlay = cv2.cvtColor(perlin_overlay, cv2.COLOR_GRAY2BGR)

    # Blend the Perlin noise with the image
    hallucination_image = cv2.addWeighted(color_shifted_image, 0.8, perlin_overlay, 0.2, 0)

    return hallucination_image
# Distort Image
def distort_image(image, noise_map, strength=20):
    height, width = image.shape[:2]
    displacement = (noise_map * strength).astype(np.float32)
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (map_x + displacement).astype(np.float32)
    map_y = (map_y + displacement).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Apply Selected Effects
def apply_selected_effects():
    if original_image is None:
        return

    scale = perlin_scale_slider.get()
    octaves = perlin_octaves_slider.get()
    persistence = perlin_persistence_slider.get() / 100
    lacunarity = perlin_lacunarity_slider.get() / 10

    processed = original_image.copy()
    height, width = processed.shape[:2]
    noise_map = generate_perlin_noise_map(width, height, scale, octaves, persistence, lacunarity)

    if active_effects["Lightning"]:
        processed = apply_lightning_effect(processed, scale, octaves, persistence, lacunarity, lightning_intensity_slider.get())
    if active_effects["Pixel Swapping"]:
        processed = apply_pixel_swapping(processed, scale, octaves, persistence, lacunarity, 0, displacement_intensity_slider.get())
    if active_effects["Marble"]:
        processed = apply_marble_texture(processed, noise_map, marble_frequency_slider.get())
    if active_effects["Waves"]:
        processed = apply_wave_texture(processed, noise_map, wave_strength_slider.get())
    if active_effects["Clouds"]:
        processed = apply_cloud_effect(processed, scale, octaves, persistence, lacunarity, cloud_opacity_slider.get() / 100)
    if active_effects["Hallucination"]:
        processed = apply_hallucination_effect(
            processed, wave_amplitude_slider.get(), wave_frequency_slider.get(), perlin_scale_slider.get(),
            color_shift_r_slider.get(), color_shift_g_slider.get(), color_shift_b_slider.get()
        )

    update_display_image(processed)

# Update Image Display
def update_display_image(image):
    global processed_image
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(processed_image))
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Load Image
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
    apply_selected_effects()

# Toggle Effects
def toggle_effect(effect_name):
    active_effects[effect_name] = not active_effects[effect_name]
    apply_selected_effects()

# GUI Setup
main_frame = tk.Frame(root, bg="#2e2e2e")
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame.pack(side="left", fill="both", expand=True)
left_frame2 = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame2.pack(side="left", fill="both", expand=True)

# Sliders
perlin_scale_slider = tk.Scale(left_frame, from_=10, to=200, label="Perlin Scale", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
perlin_scale_slider.set(perlin_scale)
perlin_scale_slider.pack(fill="x", pady=5)

perlin_octaves_slider = tk.Scale(left_frame, from_=1, to=10, label="Perlin Octaves", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
perlin_octaves_slider.set(perlin_octaves)
perlin_octaves_slider.pack(fill="x", pady=5)

perlin_persistence_slider = tk.Scale(left_frame, from_=10, to=100, label="Persistence (%)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
perlin_persistence_slider.set(int(perlin_persistence * 100))
perlin_persistence_slider.pack(fill="x", pady=5)

perlin_lacunarity_slider = tk.Scale(left_frame, from_=10, to=50, label="Lacunarity (/10)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
perlin_lacunarity_slider.set(int(perlin_lacunarity * 10))
perlin_lacunarity_slider.pack(fill="x", pady=5)

lightning_intensity_slider = tk.Scale(left_frame, from_=1, to=10, label="Lightning Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
lightning_intensity_slider.set(lightning_intensity)
lightning_intensity_slider.pack(fill="x", pady=5)

displacement_intensity_slider = tk.Scale(left_frame, from_=1, to=50, label="Displacement Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
displacement_intensity_slider.set(int(displacement_intensity * 100))
displacement_intensity_slider.pack(fill="x", pady=5)

marble_frequency_slider = tk.Scale(left_frame, from_=1, to=20, label="Marble Frequency", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
marble_frequency_slider.set(marble_frequency)
marble_frequency_slider.pack(fill="x", pady=5)

wave_strength_slider = tk.Scale(left_frame2, from_=1, to=50, label="Wave Strength", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
wave_strength_slider.set(wave_strength)
wave_strength_slider.pack(fill="x", pady=5)

cloud_opacity_slider = tk.Scale(left_frame2, from_=1, to=100, label="Cloud Opacity (%)", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
cloud_opacity_slider.set(int(cloud_opacity * 100))
cloud_opacity_slider.pack(fill="x", pady=5)

# Hallucination-specific sliders
wave_amplitude_slider = tk.Scale(left_frame2, from_=0, to=50, label="Wave Amplitude", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
wave_amplitude_slider.set(wave_amplitude)
wave_amplitude_slider.pack(fill="x", pady=5)

wave_frequency_slider = tk.Scale(left_frame2, from_=1, to=100, label="Wave Frequency", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
wave_frequency_slider.set(wave_frequency)
wave_frequency_slider.pack(fill="x", pady=5)

color_shift_r_slider = tk.Scale(left_frame2, from_=-50, to=50, label="Red Shift", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
color_shift_r_slider.set(shift_r)
color_shift_r_slider.pack(fill="x", pady=5)

color_shift_g_slider = tk.Scale(left_frame2, from_=-50, to=50, label="Green Shift", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
color_shift_g_slider.set(shift_g)
color_shift_g_slider.pack(fill="x", pady=5)

color_shift_b_slider = tk.Scale(left_frame2, from_=-50, to=50, label="Blue Shift", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_effects())
color_shift_b_slider.set(shift_b)
color_shift_b_slider.pack(fill="x", pady=5)

# Checkboxes for effects
for effect_name in active_effects:
    checkbox = tk.Checkbutton(left_frame, text=effect_name, bg="#2e2e2e", fg="white", selectcolor="#2e2e2e", command=lambda name=effect_name: toggle_effect(name))
    checkbox.pack(anchor="w", padx=10)

# Load Button
load_button = tk.Button(left_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(fill="both", pady=10)

# Right Frame for Image Display
right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
