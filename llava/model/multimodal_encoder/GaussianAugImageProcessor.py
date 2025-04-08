from transformers import CLIPImageProcessor
import numpy as np
import random
from PIL import Image
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import to_numpy_array, make_list_of_images

class CLIPImageProcessorWithNoise(CLIPImageProcessor):
    def __init__(self, *args, apply_gaussian_noise_prob=0.7, noise_std_range=(0.05, 0.15), **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_gaussian_noise_prob = apply_gaussian_noise_prob
        self.noise_std_range = noise_std_range

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Initialize the superclass using its from_pretrained method
        processor = super(CLIPImageProcessorWithNoise, cls).from_pretrained(*args, **kwargs)
        
        return processor

    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        import random
        import numpy as np

        # Randomly select the standard deviation
        std_dev = random.uniform(*self.noise_std_range)
        # Generate Gaussian noise
        noise = np.random.normal(0, std_dev, image.shape)
        # Add noise and clip values to the valid range [0, 1]
        noisy_image = np.clip(image + noise, 0, 1)
        return noisy_image

    def preprocess(self, images, apply_gaussian_noise=None, **kwargs):
        images = make_list_of_images(images)

        # If apply_gaussian_noise is not specified, use default probability
        if apply_gaussian_noise is None:
            apply_gaussian_noise = random.random() < self.apply_gaussian_noise_prob

        processed_images = []
        for image in images:
            image = to_numpy_array(image).astype(np.float32) / 255.0  # Scale to [0, 1] range
            
            if apply_gaussian_noise:
                image = self.add_gaussian_noise(image)
            
            if image.dtype != np.uint8:
                image_array = (image * 255).astype(np.uint8)  # 如果值在 [0, 1]，则将其放大到 [0, 255]
            # 将 NumPy 数组转换为 PIL.Image
            image = Image.fromarray(image_array)
    
            processed_images.append(image)
        
        # Further processing (e.g., normalization, scaling)
        return super().preprocess(processed_images, **kwargs)  # Scale back to [0, 255] range