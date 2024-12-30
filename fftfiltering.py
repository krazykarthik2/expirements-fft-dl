import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

# Utility Functions
def fft_transform(image):
    fft_image = fft2(image)
    return fftshift(fft_image)

def ifft_transform(fft_image_shifted):
    fft_image_unshifted = ifftshift(fft_image_shifted)
    return np.abs(ifft2(fft_image_unshifted))

# Filter Functions
def low_pass_filter(image, cutoff_radius):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_radius:
                mask[i, j] = 1

    filtered_fft = fft_image_shifted * mask
    return ifft_transform(filtered_fft)

def high_pass_filter(image, cutoff_radius):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_radius:
                mask[i, j] = 0

    filtered_fft = fft_image_shifted * mask
    return ifft_transform(filtered_fft)

def bandpass_filter(image, radius_low, radius_high):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if radius_low <= distance <= radius_high:
                mask[i, j] = 1

    filtered_fft = fft_image_shifted * mask
    return ifft_transform(filtered_fft)

def directional_filter(image, angle, width=10):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    y, x = y - cy, x - cx

    theta = np.arctan2(y, x)
    angle_rad = np.deg2rad(angle)

    mask = np.exp(-((theta - angle_rad) ** 2) / (2 * width ** 2))

    filtered_fft = fft_image_shifted * mask
    return ifft_transform(filtered_fft)

def anisotropic_gaussian_filter(image, sigma_diag, sigma_other):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    y, x = y - cy, x - cx

    distance_diag = (x - y) ** 2 / (2 * sigma_diag ** 2)
    distance_other = (x + y) ** 2 / (2 * sigma_other ** 2)

    filter = np.exp(-(distance_diag + distance_other))

    filtered_fft = fft_image_shifted * filter
    return ifft_transform(filtered_fft)

def ridge_frequency_enhancement(image, ridge_frequency, sigma=10):
    fft_image_shifted = fft_transform(image)
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    y, x = y - cy, x - cx

    freq_magnitude = np.sqrt(x ** 2 + y ** 2)

    filter = np.exp(-((freq_magnitude - ridge_frequency) ** 1) / (2 * sigma ** 2))

    filtered_fft = fft_image_shifted * filter
    return ifft_transform(filtered_fft)

# Example Usage
if __name__ == "__main__":
    # Load the fingerprint image (grayscale)
    fingerprint_image = cv2.imread("fingerprints/cluster_2_90.BMP", cv2.IMREAD_GRAYSCALE)

    # Apply filters
    low_pass_result = low_pass_filter(fingerprint_image, 20)
    high_pass_result = high_pass_filter(fingerprint_image, 20)
    bandpass_result = bandpass_filter(fingerprint_image, 10, 50)
    directional_result = directional_filter(fingerprint_image, angle=45, width=15)
    anisotropic_result = anisotropic_gaussian_filter(fingerprint_image, sigma_diag=5, sigma_other=10)
    ridge_result = ridge_frequency_enhancement(fingerprint_image, ridge_frequency=30, sigma=10)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 3, 1)
    plt.title("Original")
    plt.imshow(fingerprint_image, cmap="gray")

    plt.subplot(3, 3, 2)
    plt.title("Low-Pass Filter")
    plt.imshow(low_pass_result, cmap="gray")

    plt.subplot(3, 3, 3)
    plt.title("High-Pass Filter")
    plt.imshow(high_pass_result, cmap="gray")

    plt.subplot(3, 3, 4)
    plt.title("Bandpass Filter")
    plt.imshow(bandpass_result, cmap="gray")

    plt.subplot(3, 3, 5)
    plt.title("Directional Filter")
    plt.imshow(directional_result, cmap="gray")

    plt.subplot(3, 3, 6)
    plt.title("Anisotropic Gaussian Filter")
    plt.imshow(anisotropic_result, cmap="gray")

    # Function to create a circular mask
    def create_circle_mask(image_shape, radius):
        rows, cols = image_shape
        center = (rows // 2, cols // 2)
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        return mask.astype(float)

    # Function to apply FFT and circular mask
    def apply_fft_mask(image, radius):
        # Perform FFT
        fft_image = fft2(image)
        
        # Create circular mask
        mask = create_circle_mask(image.shape, radius)
        
        # Apply the mask in the frequency domain
        fft_image_shifted = fftshift(fft_image)  # Shift zero frequency component to center
        fft_image_shifted *= mask  # Multiply in the frequency domain
        
        # Inverse FFT to get the filtered image
        filtered_image = np.abs(ifft2(np.fft.ifftshift(fft_image_shifted)))  # Inverse FFT with shift
        return filtered_image

    plt.subplot(3, 3, 8)
    plt.title("Circular FFt Mask")
    plt.imshow(apply_fft_mask(fingerprint_image, 40), cmap="gray")

    plt.tight_layout()
    plt.show()