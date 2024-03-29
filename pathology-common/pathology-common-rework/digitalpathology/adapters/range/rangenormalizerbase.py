"""
This file contains the base class for normalizing image patches in a batch that were extracted from whole-slide image to a target range.
"""

#----------------------------------------------------------------------------------------------------

class RangeNormalizerBase(object):
    """
    Base class for batch normalization.
    """

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

    def process(self, patches):
        """
        Normalize the batch to the target range.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
            dict: Batch dictionary with range normalized image patches.
        """

        pass
