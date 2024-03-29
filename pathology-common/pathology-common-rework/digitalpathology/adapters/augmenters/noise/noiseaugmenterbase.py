"""
This file contains base class for augmenting patches from whole slide images with color transformations.
"""

from .. import augmenterbase as dptaugmenterbase

#----------------------------------------------------------------------------------------------------

class NoiseAugmenterBase(dptaugmenterbase.AugmenterBase):
    """Base class for noise patch augmentation."""

    def __init__(self, keyword):
        """
        Initialize the object.

        Args:
            keyword (str): Short name for the transformation.
        """

        # Initialize the base class.
        #
        super().__init__(keyword=keyword)
