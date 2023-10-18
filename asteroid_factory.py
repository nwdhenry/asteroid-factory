import os
import numpy as np
from PIL import Image, ImageFilter
from noise import snoise2
from matplotlib.path import Path

# Constants
IMAGE_SIZE = (256, 256)
ASTEROID_RADIUS = 64
NOISE_SCALE = .025
CRATER_COUNT = 50


def generate_perlin_noise_grid(size, scale=NOISE_SCALE, seed=0):
    """Generate a grid of random noise."""
    width, height = size
    noise_array = np.zeros(size)
    for x in range(width):
        for y in range(height):
            noise_array[x, y] = snoise2(x * scale, y * scale, 2, base=seed)
    return noise_array

def generate_perlin_circle(size, radius, scale=NOISE_SCALE, seed=0):
    """Generate a circular region of random noise."""
    # Generate a grid of coordinates
    width, height = size
    x = np.linspace(-radius, radius, width)
    y = np.linspace(-radius, radius, height)
    X, Y = np.meshgrid(x, y)

    # Calculate Perlin noise values for each coordinate
    noise_values = generate_perlin_noise_grid(size, scale, seed)

    # Scale the noise values to fit the circle radius
    noise_values = radius * (noise_values - np.min(noise_values)) / (np.max(noise_values) - np.min(noise_values))
    circle = np.sqrt(X ** 2 + Y ** 2) <= noise_values

    return circle.astype(int)

def generate_random_polygon(radius, num_points):
    angles = np.sort(2 * np.pi * np.random.rand(num_points))
    points = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    return points

def apply_polygon_to_mask(polygon_points, mask, shift):
    """Modify the mask according to the polygon shape."""
    # Shift polygon points to the center of the mask
    polygon_points += shift
    poly_path = Path(polygon_points)
    x, y = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    grid = poly_path.contains_points(np.column_stack([x.ravel(), y.ravel()]))
    mask[grid.reshape(mask.shape)] = 1
    return mask

def apply_perlin_noise_to_mask(perlin_noise_grid, mask):
    """Modify the mask according to the Perlin noise."""
    mask *= perlin_noise_grid
    return mask

def apply_craters(mask, crater_count, min_radius, max_radius, seed):
    """Apply impact craters to the asteroid."""
    width, height = mask.shape
    for _ in range(crater_count):
        cx, cy = np.random.randint(0, width), np.random.randint(0, height)

        crater_radius = np.random.randint(min_radius, max_radius)
        
        y, x = np.ogrid[-cx:width-cx, -cy:height-cy]
        distance_from_center = np.sqrt(x*x + y*y)
        
        # Create a gradient mask for the crater
        crater_mask = np.clip(1 - distance_from_center / crater_radius, 0, 1) * generate_perlin_circle(mask.shape, crater_radius, NOISE_SCALE / 0.1, seed)
        
        # Blend with existing mask
        mask = np.maximum(mask, mask * crater_mask)
        
    return mask

def generate_asteroid_sprite(filename, seed=None):
    """Generate the asteroid sprite."""
    if seed is None:
        seed = np.random.randint(0, 1024)
    print(f"Generating asteroid sprite {filename}...")
    print(f"Seed: {seed}")
    # Generate base mask and Perlin noise
    mask = np.zeros(IMAGE_SIZE)
    perlin_noise_grid = generate_perlin_noise_grid(IMAGE_SIZE, NOISE_SCALE, seed)
    
    # Apply random polygon shape
    polygon_points = generate_random_polygon(ASTEROID_RADIUS, np.random.randint(5, 12))
    shift = np.array([IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2])
    mask = apply_polygon_to_mask(polygon_points, mask, shift)
    # Save polygon mask for later
    poly_mask = mask.copy()
    im = Image.fromarray(np.uint8(mask * 255), "L")
    # Apply Perlin noise
    mask = apply_perlin_noise_to_mask(perlin_noise_grid, mask)
    
    # Apply craters
    min_crater_radius = int(max(min(IMAGE_SIZE) // 32, 2))
    max_crater_radius = int(min(min(IMAGE_SIZE) // 3, min(IMAGE_SIZE) / 4))
    max_crater_radius = max(max_crater_radius, min_crater_radius * 2)
    mask = apply_craters(mask, CRATER_COUNT, min_crater_radius, max_crater_radius, seed)

    # Convert to image and save
    im = Image.fromarray(np.uint8(mask * 255), "L")
    # Apply polygon mask
    im_poly = Image.fromarray(np.uint8(poly_mask * 127), "L")
    #im = Image.composite(im, im_poly, im)
    im.putalpha(im_poly)
    im = im.filter(filter=ImageFilter.BLUR)
    im.save(f"{filename}.png")

# Main function to generate multiple asteroid sprites
print(f"Generating asteroid sprites...")
dir_name = os.getcwd()
for count in range(0, 10):
    np.random.seed(count)
    generate_asteroid_sprite(filename=f"{dir_name}\\sprites\\asteroid_sprite_{count}")
    print(f"Generated asteroid sprite {count}.")

print("Done.")
