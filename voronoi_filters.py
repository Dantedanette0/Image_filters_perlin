import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import snoise2

root = tk.Tk()
root.title("Voronoi and Simplex Noise Filters")
root.geometry("1200x800")

original_image = None
processed_image = None

# Default Parameters
voronoi_points = 50  # Number of points for Voronoi cells
voronoi_seed = 42  # Seed for Voronoi generation
simplex_scale = 100  # Scale for Simplex noise
blur_intensity = 15  # Blur intensity for patch blur
distortion_intensity = 50  # Distortion intensity for channel distortion

# Active filters
active_filters = {
    "Mosaic or Stained Glass": False,
    "Patch Blur": False,
    "Channel Distortion": False,
    "Geometric Warping": False,
    "Rain Droplets": False,
    "Crystal Effect": False,
}


def generate_voronoi_map_fast(width, height, points, seed):
    np.random.seed(seed)
    point_coords = np.random.rand(points, 2) * [width, height]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_coords = np.stack((grid_x, grid_y), axis=-1)

    # Compute distances and find the closest point for each pixel
    distances = np.sum((grid_coords[:, :, None, :] - point_coords[None, None, :, :]) ** 2, axis=3)
    voronoi_map = np.argmin(distances, axis=2)
    return voronoi_map, point_coords


def generate_simplex_noise_map(width, height, scale, seed=0):
    np.random.seed(seed)
    noise_map = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            noise_map[y, x] = snoise2(x / scale, y / scale, base=seed)

    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return noise_map


def apply_mosaic_effect(image, voronoi_map):
    unique_cells = np.unique(voronoi_map)
    mosaic_image = np.zeros_like(image)

    for cell in unique_cells:
        mask = voronoi_map == cell
        average_color = cv2.mean(image, mask=mask.astype(np.uint8) * 255)[:3]
        mosaic_image[mask] = average_color

    return mosaic_image


def apply_patch_blur(image, voronoi_map, noise_map):
    blurred_image = image.copy()
    unique_cells = np.unique(voronoi_map)

    for cell in unique_cells:
        mask = (voronoi_map == cell).astype(np.uint8) * 255
        cell_indices = np.where(voronoi_map == cell)
        x_min, x_max = cell_indices[1].min(), cell_indices[1].max()
        y_min, y_max = cell_indices[0].min(), cell_indices[0].max()

        roi = image[y_min:y_max + 1, x_min:x_max + 1]
        avg_noise = np.mean(noise_map[y_min:y_max + 1, x_min:x_max + 1])
        randomness = np.random.uniform(0.8, 1.2)  # Add random factor

        kernel_size = int(avg_noise * blur_intensity_slider.get() * randomness) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        for y, x in zip(*cell_indices):
            blurred_image[y, x] = blurred_roi[y - y_min, x - x_min]

    return blurred_image


def distort_voronoi_channels(image, voronoi_map, noise_map):
    distorted_image = image.copy()
    unique_cells = np.unique(voronoi_map)

    for cell in unique_cells:
        mask = (voronoi_map == cell).astype(np.uint8) * 255
        cell_indices = np.where(voronoi_map == cell)
        x_min, x_max = cell_indices[1].min(), cell_indices[1].max()
        y_min, y_max = cell_indices[0].min(), cell_indices[0].max()

        roi = image[y_min:y_max + 1, x_min:x_max + 1]
        channel_to_distort = np.random.choice([0, 1, 2])
        noise_patch = noise_map[y_min:y_max + 1, x_min:x_max + 1]
        distortion = (noise_patch * distortion_intensity_slider.get()).astype(np.int32)

        roi[:, :, channel_to_distort] = np.clip(roi[:, :, channel_to_distort] + distortion, 0, 255)
        for y, x in zip(*cell_indices):
            distorted_image[y, x] = roi[y - y_min, x - x_min]

    return distorted_image


def apply_geometric_warping(image, voronoi_map, noise_map):
    height, width = image.shape[:2]
    warped_image = image.copy()
    unique_cells = np.unique(voronoi_map)

    for cell in unique_cells:
        mask = (voronoi_map == cell).astype(np.uint8)
        cell_indices = np.where(voronoi_map == cell)
        x_min, x_max = cell_indices[1].min(), cell_indices[1].max()
        y_min, y_max = cell_indices[0].min(), cell_indices[0].max()

        roi = image[y_min:y_max + 1, x_min:x_max + 1]
        noise_patch = noise_map[y_min:y_max + 1, x_min:x_max + 1]

        displacement_x = (noise_patch * 20).astype(np.float32)
        displacement_y = (noise_patch * 20).astype(np.float32)

        map_x, map_y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        map_x = np.clip(map_x + displacement_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(map_y + displacement_y, 0, height - 1).astype(np.float32)

        warped_roi = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for y, x in zip(*cell_indices):
            warped_image[y, x] = warped_roi[y - y_min, x - x_min]

    return warped_image


def apply_rain_droplets(image, voronoi_map, noise_map):
    height, width = image.shape[:2]
    droplets_image = image.copy()
    unique_cells = np.unique(voronoi_map)

    for cell in unique_cells:
        mask = (voronoi_map == cell).astype(np.uint8)
        cell_indices = np.where(voronoi_map == cell)
        x_min, x_max = cell_indices[1].min(), cell_indices[1].max()
        y_min, y_max = cell_indices[0].min(), cell_indices[0].max()

        roi = image[y_min:y_max + 1, x_min:x_max + 1]
        center_x, center_y = (x_max - x_min) // 2, (y_max - y_min) // 2
        radius = min(x_max - x_min, y_max - y_min) // 2

        gradient = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.float32)
        for y in range(roi.shape[0]):
            for x in range(roi.shape[1]):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                intensity = max(0, 1 - (distance / radius))
                gradient[y, x] = intensity

        gradient = (gradient * 255).astype(np.uint8)
        gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        droplets_image[y_min:y_max + 1, x_min:x_max + 1] = cv2.addWeighted(roi, 0.7, gradient, 0.3, 0)

    return droplets_image


def apply_crystal_effect(image, voronoi_map):

    height, width = image.shape[:2]
    crystal_image = image.copy()
    unique_cells = np.unique(voronoi_map)

    # Create an edge map for the Voronoi cells
    edges = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if voronoi_map[y, x] != voronoi_map[y + 1, x] or voronoi_map[y, x] != voronoi_map[y, x + 1]:
                edges[y, x] = 255

    # Dilate the edges to make them more prominent
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Process each Voronoi cell
    for cell in unique_cells:
        mask = (voronoi_map == cell).astype(np.uint8)
        cell_indices = np.where(voronoi_map == cell)
        x_min, x_max = cell_indices[1].min(), cell_indices[1].max()
        y_min, y_max = cell_indices[0].min(), cell_indices[0].max()

        # Extract the region of interest
        roi = image[y_min:y_max + 1, x_min:x_max + 1]

        # Create a shiny gradient
        center_x, center_y = (x_max - x_min) // 2, (y_max - y_min) // 2
        radius = max(x_max - x_min, y_max - y_min) // 2
        gradient = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.float32)  # Ensure single-channel gradient
        for y in range(roi.shape[0]):
            for x in range(roi.shape[1]):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                intensity = max(0, 1 - (distance / radius))
                gradient[y, x] = intensity

        # Convert gradient to proper format
        gradient = (gradient * 255).astype(np.uint8)
        gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)  # Ensure 3-channel for blending

        # Blend the gradient to simulate shininess
        shiny_roi = cv2.addWeighted(roi, 0.8, gradient, 0.2, 0)

        # Apply a slight blur to mimic glass diffusion
        blurred_roi = cv2.GaussianBlur(shiny_roi, (5, 5), 0)

        # Refraction-like warping
        noise_map = np.random.uniform(-2, 2, (roi.shape[0], roi.shape[1]))
        displacement_x = (noise_map * 2).astype(np.float32)
        displacement_y = (noise_map * 2).astype(np.float32)
        map_x, map_y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        map_x = np.clip(map_x + displacement_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(map_y + displacement_y, 0, height - 1).astype(np.float32)
        warped_roi = cv2.remap(blurred_roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Replace the cell in the crystal image
        for y, x in zip(*cell_indices):
            crystal_image[y, x] = warped_roi[y - y_min, x - x_min]

    # Highlight the edges for a sharp look
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    crystal_image = cv2.addWeighted(crystal_image, 0.9, edges_color, 0.3, 0)

    return crystal_image




def apply_selected_filters():
    if original_image is None:
        return

    num_points = voronoi_points_slider.get()
    seed_value = voronoi_seed_slider.get()

    height, width = original_image.shape[:2]
    voronoi_map, _ = generate_voronoi_map_fast(width, height, num_points, seed_value)
    noise_map = generate_simplex_noise_map(width, height, simplex_scale_slider.get(), seed=seed_value)

    processed = original_image.copy()

    if active_filters["Mosaic or Stained Glass"]:
        processed = apply_mosaic_effect(processed, voronoi_map)
    if active_filters["Patch Blur"]:
        processed = apply_patch_blur(processed, voronoi_map, noise_map)
    if active_filters["Channel Distortion"]:
        processed = distort_voronoi_channels(processed, voronoi_map, noise_map)
    if active_filters["Geometric Warping"]:
        processed = apply_geometric_warping(processed, voronoi_map, noise_map)
    if active_filters["Rain Droplets"]:
        processed = apply_rain_droplets(processed, voronoi_map, noise_map)
    if active_filters["Crystal Effect"]:
        processed = apply_crystal_effect(processed, voronoi_map)

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

button_frame = tk.Frame(left_frame, bg="#2e2e2e")
button_frame.pack(fill="x", pady=10)
load_button = tk.Button(button_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(padx=10, pady=5)

voronoi_points_slider = tk.Scale(left_frame, from_=10, to=200, label="Number of Points", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
voronoi_points_slider.set(voronoi_points)
voronoi_points_slider.pack(fill="x", pady=5)

voronoi_seed_slider = tk.Scale(left_frame, from_=0, to=100, label="Seed Value", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
voronoi_seed_slider.set(voronoi_seed)
voronoi_seed_slider.pack(fill="x", pady=5)

simplex_scale_slider = tk.Scale(left_frame, from_=10, to=200, label="Simplex Scale", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
simplex_scale_slider.set(simplex_scale)
simplex_scale_slider.pack(fill="x", pady=5)

blur_intensity_slider = tk.Scale(left_frame, from_=1, to=30, label="Blur Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
blur_intensity_slider.set(blur_intensity)
blur_intensity_slider.pack(fill="x", pady=5)

distortion_intensity_slider = tk.Scale(left_frame, from_=1, to=100, label="Distortion Intensity", orient="horizontal", bg="#2e2e2e", fg="white", command=lambda _: apply_selected_filters())
distortion_intensity_slider.set(distortion_intensity)
distortion_intensity_slider.pack(fill="x", pady=5)

for filter_name in active_filters:
    checkbox = tk.Checkbutton(left_frame, text=filter_name, bg="#2e2e2e", fg="white",selectcolor="#2e2e2e", command=lambda name=filter_name: toggle_filter(name))
    checkbox.pack(anchor="w", padx=10)

right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
