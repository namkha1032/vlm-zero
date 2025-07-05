
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]



def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample = resample, reducing_gap = reducing_gap
    )
    return resized_image



def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image*scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]]
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image
    


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample = resample) for image in images
    ]
    # Convert each image to a np array
    images = [np.array(image) for image in images]
    # rescale in range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dim since model expects input with format [channel, height, width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # in PaliGemma paper, they tokenize the \n separately, but somehow HuggingFace put it here
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    # placeholder token that will be replaced by image embedding
    IMAGE_TOKEN = "<image>"
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        # The tokenizer used here is the tokenizer of Gemma model
        # But this tokenizer was not created with the special token for the image
        # Paligemma can be used for multiple purposes (vqa, segmentation (seg token), object detection(log token))
        # But we will not use them
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}" for i in range(1024)
        ] # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<loc{i:03d}" for i in range(128)
        ] # These tokens are used for object segmentation
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        
    # this method allow an instance of the class to be call like a function
    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        # Convert the list of np arrays to a single np array with shape [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the np arrays into Pytorch tensor
        pixel_values = torch.tensor(pixel_values)
        
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        
        
        # Returns the input_ids and the attention_mask as Pytorch tensors 
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
        
        

        
                 
