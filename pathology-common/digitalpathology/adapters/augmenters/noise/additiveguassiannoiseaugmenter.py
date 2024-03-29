"""
This file contains a class for augmenting patches from whole slide images with additive Gaussian noise.
"""

from . import noiseaugmenterbase as dptnoiseaugmenterbase

from ....errors import augmentationerrors as dptaugmentationerrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class AdditiveGaussianNoiseAugmenter(dptnoiseaugmenterbase.NoiseAugmenterBase):
    """Apply additive Gaussian noise on the patch."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (tuple): Range for sigma selection for Gaussian noise. For example (0.0, 0.1).

        Raises:
            InvalidAdditiveGaussianNoiseSigmaRangeError: The sigma range for additive Gaussian noise is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='additive_gaussian_noise')

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
            sigma_range (tuple): Range for sigma selection for Gaussian noise.

        Raises:
            InvalidAdditiveGaussianNoiseSigmaRangeError: The sigma range for additive Gaussian noise is not valid.
        """

        # Check the interval.
        #
        if len(sigma_range) != 2 or sigma_range[1] < sigma_range[0] or sigma_range[0] < 0.0:
            raise dptaugmentationerrors.InvalidAdditiveGaussianNoiseSigmaRangeError(sigma_range)

        # Store the setting.
        #
        self.__sigma_range = list(sigma_range)
        self.__sigma = sigma_range[0]

    def transform(self, patch):
        """
        Apply additive Gaussian noise on the patch.

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

        # Add noise and clip the result to the valid (0.0, 1.0) range.
        #
        noise = np.random.normal(loc=0, scale=self.__sigma, size=patch.shape)
        patch_transformed = patch_normalized + noise
        patch_transformed = np.clip(a=patch_transformed, a_min=0.0, a_max=1.0)

        # Convert back to integral data type if the input was also integral.
        #
        if patch.dtype.kind != 'f':
            patch_transformed *= 255.0
            patch_transformed = patch_transformed.astype(dtype=np.uint8)

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the noise sigma.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)
