"""
This file contains a class for augmenting patches from whole slide images with Gaussian blurring.
"""

from . import noiseaugmenterbase as dptnoiseaugmenterbase

from ....errors import augmentationerrors as dptaugmentationerrors

import scipy.ndimage.filters
import numpy as np

#----------------------------------------------------------------------------------------------------

class GaussianBlurAugmenter(dptnoiseaugmenterbase.NoiseAugmenterBase):
    """Apply Gaussian blur on the patch."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (tuple): Range for sigma selection for Gaussian blur. For example (0.1, 0.5).

        Raises:
            InvalidBlurSigmaRangeError: The sigma range for Gaussian blur is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='gaussian_blur')

        # Initialize members.
        #
        self.__sigma_range = None  # Configured sigma range.
        self.__sigma = None        # Current sigma to use.

        # Save configuration.
        #
        self.__setsigmarange(sigma_range=sigma_range)

    def __setsigmarange(self, sigma_range):
        """
        Set the sigma range.

        Args:
            sigma_range (tuple): Range for sigma selection for Gaussian blur.

        Raises:
            InvalidBlurSigmaRangeError: The sigma range for Gaussian blur is not valid.
        """

        # Check the interval.
        #
        if len(sigma_range) != 2 or sigma_range[1] < sigma_range[0] or sigma_range[0] < 0.0:
            raise dptaugmentationerrors.InvalidBlurSigmaRangeError(sigma_range)

        # Store the setting.
        #
        self.__sigma_range = list(sigma_range)
        self.__sigma = sigma_range[0]

    def transform(self, patch):
        """
        Blur the patch with a random sigma.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Normalize patch range to [0.0, 1.0].
        #
        if patch.dtype.kind == 'f':
            patch_normalized = patch
        else:
            patch_normalized = patch.astype(dtype=np.float32) / 255.0

        # Blur the patch by channels.
        #
        patch_transformed = scipy.ndimage.filters.gaussian_filter(input=patch_normalized, sigma=(self.__sigma, self.__sigma, 0.0))

        # Convert back to integral data type if the input was also integral.
        #
        if patch.dtype.kind != 'f':
            patch_transformed *= 255.0
            patch_transformed = patch_transformed.astype(dtype=np.uint8)

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)
