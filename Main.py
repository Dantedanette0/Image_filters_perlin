import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from noise import pnoise2, snoise2
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree

root = tk.Tk()
root.title("Advanced Image Noise Adjustment")
root.geometry("1400x800") 

# noise control
apply_gaussian = tk.BooleanVar(master=root, value=False)
apply_salt_pepper = tk.BooleanVar(master=root, value=False)
apply_perlin = tk.BooleanVar(master=root, value=False)
apply_simplex = tk.BooleanVar(master=root, value=False)
apply_voronoi = tk.BooleanVar(master=root, value=False)
vornoi_strong = tk.BooleanVar(master=root, value=False)
perlin_strong = tk.BooleanVar(master=root, value=False)
simplex_strong = tk.BooleanVar(master=root, value=False)

# mode selection 
displacement_mode = tk.BooleanVar(master=root, value=False)
displacement_distance = 10
# channel selection
apply_red = tk.BooleanVar(master=root, value=True)
apply_green = tk.BooleanVar(master=root, value=True)
apply_blue = tk.BooleanVar(master=root, value=True)

# images
original_image = None
processed_image = None

# Perlin Noise
perlin_scale = 100
perlin_octaves = 1
perlin_persistence = 0.5
perlin_lacunarity = 2.0


# Simplex Noise
simplex_scale = 100
simplex_octaves = 1
simplex_persistence = 0.5
simplex_lacunarity = 2.0



# Voronoi Noise
voronoi_num_points = 100
voronoi_height = 0
voronoi_width = 0


# Gaussian noise
gaussian_mean = 0
gaussian_std_dev = 25

# salt and pepper noise
salt_prob = 0.0
pepper_prob = 0.0

# updates a given global variables and updates the image with the new variables
def update_global_and_refresh(var_name, value):
    globals()[var_name] = value
    update_image()

def generate_perlin_noise_mask(width, height, scale, octaves, persistence, lacunarity, seed=0):
    np.random.seed(seed)
    perlin_noise = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            perlin_noise[y][x] = pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_simplex_noise_mask(width, height, scale, octaves, persistence, lacunarity, seed=0):
    np.random.seed(seed)
    simplex_noise = np.zeros((height,width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            simplex_noise[y][x] = snoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
    simplex_noise = (simplex_noise - np.min(simplex_noise)) / (np.max(simplex_noise) - np.min(simplex_noise))
    return simplex_noise

def generate_voronoi_noise_mask(width, height, num_points, seed=0):
    np.random.seed(seed)
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)

    voronoi_image = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            distances = np.linalg.norm(points - np.array([x, y]), axis=1)
            voronoi_image[y, x] = np.min(distances)

    voronoi_image = (voronoi_image - np.min(voronoi_image)) / (np.max(voronoi_image) - np.min(voronoi_image))
    return voronoi_image

def apply_perlin_noise(image, scale, octaves, persistence, lacunarity):
    height, width = image.shape[:2]
    perlin_noise = generate_perlin_noise_mask(width, height, scale, octaves, persistence, lacunarity)
    perlin_noise = (perlin_noise * 255).astype(np.uint8)

    noisy_image = image.copy()
    if perlin_strong.get():
            if apply_red.get():
                noisy_image[:, :, 2] = perlin_noise 
            if apply_green.get():
                noisy_image[:, :, 1] = perlin_noise  
            if apply_blue.get():
                noisy_image[:, :, 0] = perlin_noise 
    else:
        if apply_red.get():
            noisy_image[:, :, 2] = cv2.addWeighted(noisy_image[:, :, 2], 0.7, perlin_noise, 0.3, 0)
        if apply_green.get():
            noisy_image[:, :, 1] = cv2.addWeighted(noisy_image[:, :, 1], 0.7, perlin_noise, 0.3, 0)
        if apply_blue.get():
            noisy_image[:, :, 0] = cv2.addWeighted(noisy_image[:, :, 0], 0.7, perlin_noise, 0.3, 0)
    return noisy_image

def apply_simplex_noise(image, scale, octaves, persistence, lacunarity):
    height, width = image.shape[:2]
    simplex_noise = generate_simplex_noise_mask(width, height, scale, octaves, persistence, lacunarity)
    simplex_noise = (simplex_noise * 255).astype(np.uint8)
    noisy_image = image.copy()
    if simplex_strong.get():
        if apply_red.get():
            noisy_image[:, :, 2] = simplex_noise 
        if apply_green.get():
            noisy_image[:, :, 1] = simplex_noise  
        if apply_blue.get():
            noisy_image[:, :, 0] = simplex_noise  
    else:
        if apply_red.get():
            noisy_image[:, :, 2] = cv2.addWeighted(noisy_image[:, :, 2], 0.7, simplex_noise, 0.3, 0)
        if apply_green.get():
            noisy_image[:, :, 1] = cv2.addWeighted(noisy_image[:, :, 1], 0.7, simplex_noise, 0.3, 0)
        if apply_blue.get():
            noisy_image[:, :, 0] = cv2.addWeighted(noisy_image[:, :, 0], 0.7, simplex_noise, 0.3, 0)
    return noisy_image


def apply_voronoi_noise(image, num_points):
    height, width = image.shape[:2]
    voronoi_noise = generate_voronoi_noise_mask(width, height, num_points)
    voronoi_noise = (voronoi_noise * 255).astype(np.uint8)

    noisy_image = image.copy()
    if vornoi_strong.get():
        if apply_red.get():
            noisy_image[:, :, 2] = voronoi_noise
        if apply_green.get():
            noisy_image[:, :, 1] = voronoi_noise
        if apply_blue.get():
            noisy_image[:, :, 0] = voronoi_noise
    else:
        if apply_red.get():
            noisy_image[:, :, 2] = cv2.addWeighted(noisy_image[:, :, 2], 0.7, voronoi_noise, 0.3, 0)
        if apply_green.get():
            noisy_image[:, :, 1] = cv2.addWeighted(noisy_image[:, :, 1], 0.7, voronoi_noise, 0.3, 0)
        if apply_blue.get():
            noisy_image[:, :, 0] = cv2.addWeighted(noisy_image[:, :, 0], 0.7, voronoi_noise, 0.3, 0)
    return noisy_image

def add_gaussian_noise(image, mean, std_dev):
    noisy_image = image.astype(np.float32)
    if apply_red.get():
        noisy_image[:, :, 2] += np.random.normal(mean, std_dev, image[:, :, 2].shape)
    if apply_green.get():
        noisy_image[:, :, 1] += np.random.normal(mean, std_dev, image[:, :, 1].shape)
    if apply_blue.get():
        noisy_image[:, :, 0] += np.random.normal(mean, std_dev, image[:, :, 0].shape)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    if apply_red.get():
        noisy_image[:, :, 2][salt_mask] = 255
        noisy_image[:, :, 2][pepper_mask] = 0
    if apply_green.get():
        noisy_image[:, :, 1][salt_mask] = 255
        noisy_image[:, :, 1][pepper_mask] = 0
    if apply_blue.get():
        noisy_image[:, :, 0][salt_mask] = 255
        noisy_image[:, :, 0][pepper_mask] = 0
    return noisy_image

def generate_perlin_displacement_map(width, height, scale, octaves, persistence, lacunarity, seed):
    displacement_x = np.zeros((height, width), dtype=np.float32)
    displacement_y = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            displacement_x[y, x] = pnoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )
            displacement_y[y, x] = pnoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 1
            )

    displacement_x = (displacement_x - np.min(displacement_x)) / (np.max(displacement_x) - np.min(displacement_x))
    displacement_y = (displacement_y - np.min(displacement_y)) / (np.max(displacement_y) - np.min(displacement_y))

    displacement_x = (displacement_x * width * displacement_distance).astype(np.int32)
    displacement_y = (displacement_y * height * displacement_distance).astype(np.int32)

    return displacement_x, displacement_y


def apply_perlin_pixel_swapping(image, scale, octaves, persistence, lacunarity):
    seed = 0
    np.random.seed(seed)
    height, width = image.shape[:2]
    displacement_x, displacement_y = generate_perlin_displacement_map(width, height, scale, octaves, persistence, lacunarity, seed)
    swapped_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            swapped_image[y, x] = image[new_y, new_x]

    return swapped_image

def add_salt_and_pepper_displacement(image, salt_prob, pepper_prob):

    noisy_image = np.copy(image)
    height, width = image.shape[:2]

    salt_mask = np.random.rand(height, width) < salt_prob
    pepper_mask = np.random.rand(height, width) < pepper_prob

    for y in range(height):
        for x in range(width):
            if salt_mask[y, x]:
                new_y = int((y+y * displacement_distance) % height)
                if apply_red:
                    noisy_image[y, x, 2] = image[new_y, x, 2]  
                if apply_green:
                    noisy_image[y, x, 1] = image[new_y, x, 1]  
                if apply_blue:
                    noisy_image[y, x, 0] = image[new_y, x, 0]  

            if pepper_mask[y, x]:
                new_x = int((x+ x * displacement_distance) % width)
                if apply_red:
                    noisy_image[y, x, 2] = image[y, new_x, 2]  
                if apply_green:
                    noisy_image[y, x, 1] = image[y, new_x, 1]  
                if apply_blue:
                    noisy_image[y, x, 0] = image[y, new_x, 0] 

    return noisy_image

def generate_voronoi_displacement_map(width, height, points, intensity):
    # Generate random points
    np.random.seed(0)
    point_coords = np.random.rand(points, 2) * [width, height]

    # Build a KDTree for distance queries
    tree = cKDTree(point_coords)

    displacement_x = np.zeros((height, width), dtype=np.int32)
    displacement_y = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            _, idx = tree.query([x, y])
            nearest_point = point_coords[idx]
            displacement = (nearest_point - [x, y]) * intensity
            displacement_x[y, x] = int(displacement[0])
            displacement_y[y, x] = int(displacement[1])

    return displacement_x, displacement_y


def apply_voronoi_pixel_swapping(image, points):
    height, width = image.shape[:2]
    displacement_x, displacement_y = generate_voronoi_displacement_map(width, height, points, displacement_distance)

    swapped_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            swapped_image[y, x] = image[new_y, new_x]

    return swapped_image
def generate_simplex_displacement_map(width, height, scale, octaves, persistence, lacunarity, seed):
    displacement_x = np.zeros((height, width), dtype=np.float32)
    displacement_y = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            displacement_x[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed
            )
            displacement_y[y, x] = snoise2(
                x / scale, y / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed + 1
            )

    displacement_x = (displacement_x - np.min(displacement_x)) / (np.max(displacement_x) - np.min(displacement_x))
    displacement_y = (displacement_y - np.min(displacement_y)) / (np.max(displacement_y) - np.min(displacement_y))

    displacement_x = (displacement_x * width * displacement_distance).astype(np.int32)
    displacement_y = (displacement_y * height * displacement_distance).astype(np.int32)

    return displacement_x, displacement_y


def apply_simplex_pixel_swapping(image, scale, octaves, persistence, lacunarity, seed=0):
    height, width = image.shape[:2]
    displacement_x, displacement_y = generate_simplex_displacement_map(width, height, scale, octaves, persistence, lacunarity, seed)

    swapped_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            new_x = (x + displacement_x[y, x]) % width
            new_y = (y + displacement_y[y, x]) % height
            swapped_image[y, x] = image[new_y, new_x]

    return swapped_image


def add_gaussian_displacement(image, mean, std_dev):

    height, width = image.shape[:2]
    
    # Generate Gaussian noise for x and y displacements
    x_displacement = (np.random.normal(mean, std_dev, (height, width)) * displacement_distance).astype(np.float32)
    y_displacement = (np.random.normal(mean, std_dev, (height, width)) * displacement_distance).astype(np.float32)
    
    # Create remap grids
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (map_x + x_displacement).astype(np.float32)
    map_y = (map_y + y_displacement).astype(np.float32)
    
    # Apply remapping
    displaced_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return displaced_image



def update_image():
    if original_image is None:
        return
    noisy_image = original_image.copy()
    if displacement_mode.get():
        if apply_gaussian.get():
            noisy_image = add_gaussian_displacement(noisy_image, gaussian_mean, gaussian_std_dev)

        if apply_salt_pepper.get():
            noisy_image = add_salt_and_pepper_displacement(noisy_image, salt_prob, pepper_prob)

        if apply_perlin.get():
            noisy_image = apply_perlin_pixel_swapping(noisy_image, perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity)

        if apply_simplex.get():
            noisy_image = apply_simplex_pixel_swapping(noisy_image,simplex_scale, simplex_octaves, simplex_persistence, simplex_lacunarity)

        if apply_voronoi.get():
            noisy_image = apply_voronoi_pixel_swapping(noisy_image, voronoi_num_points)

    else:
        if apply_gaussian.get():
            noisy_image = add_gaussian_noise(noisy_image, gaussian_mean, gaussian_std_dev)

        if apply_salt_pepper.get():
            noisy_image = add_salt_and_pepper_noise(noisy_image, salt_prob, pepper_prob)

        if apply_perlin.get():
            noisy_image = apply_perlin_noise(noisy_image, perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity)

        if apply_simplex.get():
            noisy_image = apply_simplex_noise(noisy_image,simplex_scale, simplex_octaves, simplex_persistence, simplex_lacunarity)

        if apply_voronoi.get():
            noisy_image = apply_voronoi_noise(noisy_image, voronoi_num_points)

    update_display_image(noisy_image)


def update_display_image(image):
    global processed_image
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(processed_image))
    image_label.config(image=img_tk)
    image_label.image = img_tk

def create_slider(frame, label_text, from_, to_, command):
    label = tk.Label(frame, text=label_text, font=("Arial", 12, "bold"), fg="white", bg="#2e2e2e")
    label.pack(anchor="w", padx=10)
    slider = ttk.Scale(frame, from_=from_, to_=to_, orient="horizontal", command=command, style="Custom.Horizontal.TScale")
    slider.set((from_ + to_) / 2)
    slider.pack(fill="x", padx=10, pady=5)
    return slider

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

    update_image()

# Configure styles
style = ttk.Style(root)
style.theme_use("default")
style.configure("Custom.Horizontal.TScale", troughcolor="#4e4e4e", sliderthickness=15, background="#2e2e2e")

# Main Layout
main_frame = tk.Frame(root, bg="#2e2e2e")
main_frame.pack(fill="both", expand=True)

# Left Section: Controls
left_frame = tk.Frame(main_frame, bg="#2e2e2e", padx=10, pady=10)
left_frame.pack(side="left", fill="y", expand=True)

# Load Image Button
button_frame = tk.Frame(left_frame, bg="#2e2e2e")
button_frame.pack(fill="x", pady=10)
load_button = tk.Button(button_frame, text="Load Image", command=load_image, bg="#4e4e4e", fg="white")
load_button.pack(padx=10)

# Channel Checkboxes
checkbox_frame = tk.Frame(left_frame, bg="#2e2e2e", padx=10, pady=10)
checkbox_frame.pack(fill="x")
tk.Checkbutton(checkbox_frame, text="Red", variable=apply_red, bg="#ff4242", fg="black", selectcolor="white", command=update_image).pack(side="left", padx=10)
tk.Checkbutton(checkbox_frame, text="Green", variable=apply_green, bg="#42ff42", fg="black", selectcolor="white", command=update_image).pack(side="left", padx=10)
tk.Checkbutton(checkbox_frame, text="Blue", variable=apply_blue, bg="#42d3ff", fg="black", selectcolor="white", command=update_image).pack(side="left", padx=10)

#displacement mode controls 
tk.Checkbutton(checkbox_frame, text="Displacement Mode", variable=displacement_mode, bg="#b1b3b2", fg="black", selectcolor="white", command=update_image).pack(side="left", padx=10)
create_slider(checkbox_frame, "Displacement Factor", 0.01, 5, lambda v: update_global_and_refresh("displacement_distance", float(v)))

# Two Columns for Controls
controls_frame = tk.Frame(left_frame, bg="#2e2e2e")
controls_frame.pack(fill="both", expand=True)

col1_frame = tk.Frame(controls_frame, bg="#2e2e2e")
col1_frame.pack(side="left", fill="both", expand=True)

col2_frame = tk.Frame(controls_frame, bg="#2e2e2e")
col2_frame.pack(side="right", fill="both", expand=True)

# Gaussian Noise Controls
gaussian_frame = tk.LabelFrame(col1_frame, text="Gaussian Noise Controls", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
gaussian_frame.pack(fill="x", pady=50,padx=10)
tk.Checkbutton(gaussian_frame, text="Enable Gaussian Noise", variable=apply_gaussian, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="w", padx=10)
create_slider(gaussian_frame, "Gaussian Mean", -50, 50, lambda v: update_global_and_refresh("gaussian_mean", float(v)))
create_slider(gaussian_frame, "Gaussian Std Dev", 1, 100, lambda v: update_global_and_refresh("gaussian_std_dev", float(v)))

# Salt and Pepper Noise Controls
sp_frame = tk.LabelFrame(col2_frame, text="Salt & Pepper Noise Controls", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
sp_frame.pack(fill="x", pady=10)
tk.Checkbutton(sp_frame, text="Enable Salt & Pepper Noise", variable=apply_salt_pepper, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="w", padx=10)
create_slider(sp_frame, "Salt Probability", 0.0, 1, lambda v: update_global_and_refresh("salt_prob", float(v)))
create_slider(sp_frame, "Pepper Probability", 0.0, 1, lambda v: update_global_and_refresh("pepper_prob", float(v)))

# Perlin Noise Controls
perlin_frame = tk.LabelFrame(col1_frame, text="Perlin Noise Controls", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
perlin_frame.pack(fill="x", pady=10,padx=10)
tk.Checkbutton(perlin_frame, text="Enable Perlin Noise", variable=apply_perlin, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="nw", padx=10)
tk.Checkbutton(perlin_frame, text="Strong", variable=perlin_strong, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="nw", padx=10)
create_slider(perlin_frame, "Perlin Scale", 10, 200, lambda v: update_global_and_refresh("perlin_scale", int(float(v))))
create_slider(perlin_frame, "Perlin Octaves", 1, 10, lambda v: update_global_and_refresh("perlin_octaves", int(float(v))))
create_slider(perlin_frame, "Perlin Persistence", 0.1, 1.0, lambda v: update_global_and_refresh("perlin_persistence", float(v)))
create_slider(perlin_frame, "Perlin Lacunarity", 1.0, 5.0, lambda v: update_global_and_refresh("perlin_lacunarity", float(v)))

# Simplex Noise Controls
simplex_frame = tk.LabelFrame(col2_frame, text="Simplex Noise Controls", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
simplex_frame.pack(fill="x", pady=10)
tk.Checkbutton(simplex_frame, text="Enable Simplex Noise", variable=apply_simplex, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="w", padx=10)
tk.Checkbutton(simplex_frame, text="Strong", variable=simplex_strong, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="nw", padx=10)
create_slider(simplex_frame, "Simplex Scale", 10, 200, lambda v: update_global_and_refresh("simplex_scale", int(float(v))))
create_slider(simplex_frame, "Simplex Octaves", 1, 10, lambda v: update_global_and_refresh("simplex_octaves", int(float(v))))
create_slider(simplex_frame, "Simplex Persistence", 0.1, 1.0, lambda v: update_global_and_refresh("simplex_persistence", float(v)))
create_slider(simplex_frame, "Simplex Lacunarity", 1.0, 5.0, lambda v: update_global_and_refresh("simplex_lacunarity", float(v)))

# Voronoi Noise Controls
voronoi_frame = tk.LabelFrame(col2_frame, text="Voronoi Noise Controls", bg="#2e2e2e", fg="white", font=("Arial", 14, "bold"))
voronoi_frame.pack(fill="x", pady=10)
tk.Checkbutton(voronoi_frame, text="Enable Voronoi Noise", variable=apply_voronoi, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="nw", padx=10)
tk.Checkbutton(voronoi_frame, text="Strong", variable=vornoi_strong, bg="#2e2e2e", fg="white", selectcolor="#4e4e4e", command=update_image).pack(anchor="nw", padx=10)
create_slider(voronoi_frame, "Voronoi Points", 10, 50, lambda v: update_global_and_refresh("voronoi_num_points", int(float(v))))


# Image Display Frame
right_frame = tk.Frame(main_frame, bg="#4e4e4e", bd=5, relief="ridge")
right_frame.pack(side="right", fill="both", expand=True)
image_label = tk.Label(right_frame, bg="#2e2e2e")
image_label.pack()

root.mainloop()
