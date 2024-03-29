"""
This file contains a class for augmenting patches from whole slide images with left-right or upside-down flipping.
"""

from . import spatialaugmenterbase as dptspatialaugmenterbase

from ....errors import augmentationerrors as dptaugmentationerrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class FlipAugmenter(dptspatialaugmenterbase.SpatialAugmenterBase):
    """Mirrors patch vertically, horizontally or both."""

    def __init__(self, flip_list):
        """
        Initialize the object.

        Args:
            flip_list (list): List of possible flips. Example: flip_list = ['none', 'vertical', 'horizontal', 'both'].

        Raises:
            InvalidFlipListError: The flip list is invalid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='flip')

        # Initialize members.
        #
        self.__flip_list = []  # List of possible flip modes.
        self.__flip = None     # Current flip to use.

        # Save configuration.
        #
        self.__setfliplist(flip_list=flip_list)

    def __setfliplist(self, flip_list):
        """
        Save the flip direction set.

        Args:
            flip_list (list): List of possible flips. Example: flip_list = ['none', 'vertical', 'horizontal', 'both'].

        Raises:
            InvalidFlipListError: The flip list is invalid.
        """

        # Check the list.
        #
        if not set(flip_list) <= {'none', 'vertical', 'horizontal', 'both'}:
            raise dptaugmentationerrors.InvalidFlipListError(flip_list)

        # Store the setting.
        #
        self.__flip_list = flip_list
        self.__flip = self.__flip_list[0]

    def transform(self, patch):
        """
        Flip the given patch none, vertically, horizontally or both.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Flip the patch.
        #
        if self.__flip == 'none':
            patch_transformed = patch
        elif self.__flip == 'vertical':
            patch_transformed = np.flipud(patch)
        elif self.__flip == 'horizontal':
            patch_transformed = np.fliplr(patch)
        elif self.__flip == 'both':
            patch_transformed = np.fliplr(np.flipud(patch))
        else:
            raise dptaugmentationerrors.InvalidFlipMode(self.__flip)

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the flip direction.
        #
        self.__flip = np.random.choice(a=self.__flip_list, size=None)
