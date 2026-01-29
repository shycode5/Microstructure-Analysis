import numpy as np
import cv2
import os

def generate_synthetic_microstructure(filename="test_structure.tif", size=(512, 512)):
    # Create random blobs for Pearlite (Dark) and background Ferrite (Light)
    img = np.full(size, 200, dtype=np.uint8) # Light background (Ferriteish)
    
    # Generate random noise
    noise = np.random.normal(0, 20, size)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add Pearlite grains (Dark blobs)
    num_blobs = 30
    for _ in range(num_blobs):
        center = np.random.randint(0, size[0], 2)
        radius = np.random.randint(20, 60)
        cv2.circle(img, tuple(center), radius, (50), -1)
        
    # Smooth to make it look organic
    img = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Add more noise
    final_noise = np.random.normal(0, 5, size).astype(np.uint8)
    img = cv2.add(img, final_noise)
    
    cv2.imwrite(filename, img)
    print(f"Synthetic image saved to {filename}")

if __name__ == "__main__":
    generate_synthetic_microstructure()
