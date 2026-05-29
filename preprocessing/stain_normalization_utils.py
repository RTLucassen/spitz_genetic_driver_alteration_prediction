"""
Utility functions for stain normalization.

Implementation adapted from the Macenko stain normalization of the TIA Toolbox.

Changes:
    - Transform implementation can now take the target stain matrix and maximum 
      concentration as arguments.
"""

import cv2
import numpy as np
from skimage import exposure

def get_stain_matrix( 
    img: np.ndarray, 
    luminosity_threshold: float = 0.8, 
    angular_percentile: float = 99,
) -> np.ndarray:
    """Macenko stain extractor.

    Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
    img (:class:`numpy.ndarray`):
        Input image used for stain matrix estimation.
    luminosity_threshold (float):
        Threshold used for tissue area selection
    angular_percentile (int):
        Percentile of angular coordinates to be selected
        with respect to the principle, orthogonal eigenvectors.

    Returns:
        :class:`numpy.ndarray`:
            Estimated stain matrix.

    """
    img = img.astype("uint8")  # ensure input image is uint8

    # convert to OD and ignore background
    tissue_mask = get_luminosity_tissue_mask(
        img,
        threshold=luminosity_threshold,
    ).reshape((-1,))
    img_od = rgb2od(img).reshape((-1, 3))
    img_od = img_od[tissue_mask]

    # eigenvectors of covariance in OD space (orthogonal as covariance symmetric)
    _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))

    # the two principle eigenvectors
    eigen_vectors = eigen_vectors[:, [2, 1]]

    # make sure vectors are pointing the right way
    eigen_vectors = vectors_in_correct_direction(e_vectors=eigen_vectors)

    # project on this basis.
    proj = np.dot(img_od, eigen_vectors)

    # angular coordinates with respect to the principle, orthogonal eigenvectors
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    # min and max angles
    min_phi = np.percentile(phi, 100 - angular_percentile)
    max_phi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))

    # order of H&E - H first row
    he = h_and_e_in_right_order(v1, v2)

    return he / np.linalg.norm(he, axis=1)[:, None]


def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Estimate concentration matrix given an image and stain matrix.

    Args:
        img (:class:`numpy.ndarray`):
            Input image.
        stain_matrix (:class:`numpy.ndarray`):
            Stain matrix for haematoxylin and eosin stains.

    Returns:
        numpy.ndarray:
            Stain concentrations of input image.

    """
    od = rgb2od(img).reshape((-1, 3))
    x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
    return x.T


def transform(
    img: np.ndarray, 
    stain_matrix_target:np.ndarray, 
    maxC_target: np.ndarray,
) -> np.ndarray:
    """Transform an image.

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB input source image.

    Returns:
        :class:`numpy.ndarray`:
            RGB stain normalized image.

    """
    stain_matrix_source = get_stain_matrix(img)
    source_concentrations = get_concentrations(img, stain_matrix_source)
    max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    source_concentrations *= maxC_target / max_c_source
    trans = 255 * np.exp(
        -1 * np.dot(source_concentrations, stain_matrix_target),
    )

    # ensure between 0 and 255
    trans[trans > 255] = 255  # noqa: PLR2004
    trans[trans < 0] = 0

    return trans.reshape(img.shape).astype(np.uint8)


# define helper functions
def vectors_in_correct_direction(e_vectors: np.ndarray) -> np.ndarray:
    """Points the eigen vectors in the right direction.

    Args:
        e_vectors (:class:`numpy.ndarray`):
            Eigen vectors.

    Returns:
        :class:`numpy.ndarray`:
            Pointing in the correct direction.

    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Rearrange input vectors for H&E in correct order with H as first output.

    Args:
        v1 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`):
            Input vector for stain extraction.

    Returns:
        :class:`numpy.ndarray`:
            Input vectors in the correct order.

    """
    if v1[0] > v2[0]:
        return np.array([v1, v2])

    return np.array([v2, v1])


def dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    """Return correct value for H and E from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        :class:`numpy.ndarray`:
            With correct values for H and E.

    """
    if dictionary[0, 0] < dictionary[1, 0]:
        return dictionary[[1, 0], :]

    return dictionary


def rgb2od(img: np.ndarray) -> np.ndarray:
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.

    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)

    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od: np.ndarray) -> np.ndarray:
    r"""Convert from optical density (:math:`OD_{RGB}`) to RGB.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        od (:class:`numpy.ndarray`):
            Optical density (OD) RGB image.

    Returns:
        :class:`numpy.ndarray`:
            RGB Image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)

    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def contrast_enhancer(img: np.ndarray, low_p: int = 2, high_p: int = 98) -> np.ndarray:
    """Enhance contrast of the input image using intensity adjustment.

    This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`):
            Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    """
    # check if image is not uint8
    if img.dtype != np.uint8:
        msg = "Image should be uint8."
        raise AssertionError(msg)
    img_out = img.copy()
    percentiles = np.array(np.percentile(img_out, (low_p, high_p)))
    p_low, p_high = percentiles[0], percentiles[1]
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out,
            in_range=(p_low, p_high),
            out_range=(0.0, 255.0),
        )
    return img_out.astype(np.uint8)


def get_luminosity_tissue_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.

    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)

    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    return tissue_mask
