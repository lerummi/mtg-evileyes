import os
import glob
from PIL import Image
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm
from typing import List, Generator, Optional


def compute_patches(size: int, patch_size: int, window_shift: int) -> List[slice]:
    """
    Given images of different sizes, compute the patches of the image.

    :param size: Size of the image
    :param patch_size: Size of the patch
    :param window_shift: Shift of the window (i.e. the shift of the window for 
                         subsequent patches)
    
    :return: List of slices of the patches
    """

    patch_overlap = patch_size - window_shift

    if patch_overlap >= patch_size:
        raise ValueError("patch_overlap must be less than patch_size")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    elif size <= patch_size:
        patches = []
    elif size <= patch_size + patch_overlap:
        patches = [slice(0, patch_size), slice(size - patch_size, size)]
    else:
        n_patches_minus_1 = ((size - patch_size) // patch_overlap) + 1
        #print(n_patches_minus_1)
        # Shift of window
        delta = (size - patch_size) / n_patches_minus_1
        #print(delta)
        patches = []
        n_patches = n_patches_minus_1 + 1
        for i in range(n_patches):
            start_at = int(i * delta)
            end_at = int(i * delta + patch_size)
            patches.append(slice(start_at, end_at))

    return patches


def get_images(
        source_pattern: str, 
        zoom: float = 1, 
        skip: int = None
    ) -> Generator:
    """
    Given a pattern, get the images from the source.
    
    :param source_pattern: Pattern of the source
    :param zoom: Zoom factor
    :param skip: Number of images to skip

    :return: Generator of images
    """

    files = list(glob.glob(source_pattern))
    if skip is not None:
        files = files[skip:]

    for file in tqdm(files, total=len(files)):

        uuid = os.path.splitext(os.path.basename(file))[0]
        info = {"zoom": zoom}
        q = Image.open(file)
        
        if zoom != 1:
            q = q.resize((int(zoom * q.size[0] + .5), int(zoom * q.size[1] + .5)))

        yield uuid, info, np.array(q)


def get_image_patches(
        source_pattern: str, 
        patch_size: int = 224,
        window_shift: int = 56,
        batch_size: int = 128,
        skip: int = None,
        zoom: float = 1
        ) -> Generator:
    """

    Given a pattern, get the patches from the source.

    :param source_pattern: Pattern of the source
    :param patch_size: Size of the patch
    :param window_shift: Shift of the window (i.e. the shift of the window for
                         subsequent patches)
    :param batch_size: Size of the batch
    :param skip: Number of images to skip
    :param zoom: Zoom factor
    
    :return: Generator of patches
    """

    patches_uuid = []
    patches_info = []
    images = []

    image_generator = get_images(source_pattern=source_pattern, skip=skip, zoom=zoom)
    
    for uuid, info, q in image_generator:
        image_size = q.shape

        px = compute_patches(image_size[0], patch_size, window_shift)
        py = compute_patches(image_size[1], patch_size, window_shift)

        for px_ in px:
            for py_ in py:
                patches_uuid.append(uuid)
                patch_info = deepcopy(info)
                patch_info.update(
                    {
                        "xstart": px_.start,
                        "xstop": px_.stop,
                        "ystart": py_.start,
                        "ystop": py_.stop,
                    }
                )
                patches_info.append(patch_info)

                images.append(q[px_, py_, :])

                if len(images) == batch_size:
                    yield np.array(patches_uuid), np.array(patches_info), np.array(images)
                    patches_uuid = []
                    patches_info = []
                    images = []

    yield patches_uuid, patches_info, np.array(images)
