"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2


def write_images(images, prefix='test'):
    for i, image in enumerate(images):
        cv2.imwrite("{}{}.png".format(prefix, i), normalize_and_scale(image))


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1/8)



def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1/8)


def uniform(img, ksize):
    return cv2.boxFilter(img, -1, ksize)


def applyUniform(Ix, Iy, It, ksize):
    SIxIx = uniform(Ix * Ix, ksize)
    SIxIy = uniform(Ix * Iy, ksize)
    SIyIx = uniform(Iy * Ix, ksize)
    SIyIy = uniform(Iy * Iy, ksize)
    a_arrays = (SIxIx, SIxIy, SIyIx, SIyIy)

    SIxIt = uniform(Ix * It, ksize)
    SIyIt = uniform(Iy * It, ksize)
    b_arrays = (-SIxIt, -SIyIt)

    return a_arrays, b_arrays


def gausian(img, ksize, sigma):
    return cv2.GaussianBlur(img, ksize, sigma)


def applyGausian(Ix, Iy, It, ksize, sigma):
    SIxIx = gausian(Ix * Ix, ksize, sigma)
    SIxIy = gausian(Ix * Iy, ksize, sigma)
    SIyIx = gausian(Iy * Ix, ksize, sigma)
    SIyIy = gausian(Iy * Iy, ksize, sigma)
    a_arrays = (SIxIx, SIxIy, SIyIx, SIyIy)

    SIxIt = gausian(Ix * It, ksize, sigma)
    SIyIt = gausian(Iy * It, ksize, sigma)
    b_arrays = (-SIxIt, -SIyIt)

    return a_arrays, b_arrays


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    if k_type == '':
        k_type = 'uniform'

    # Validate the provided type
    assert k_type in ['uniform', 'gaussian'], \
        "Invlid k_type: '{}'".format(k_type)

    # Compute the grandients of the image
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a

    # Apply the corresponding filter
    ksize = (k_size, k_size)
    if k_type == 'uniform':
        a_arrays, b_arrays = applyUniform(Ix,Iy, It, ksize)
    elif k_type == 'gaussian':
        a_arrays, b_arrays = applyGausian(Ix,Iy, It, ksize, sigma)

    # Set the a and b matrices to be solved.
    shape = img_a.shape
    a = np.stack(a_arrays, axis=2).reshape(shape[0], shape[1], 2, 2)
    b = np.stack(b_arrays, axis=2)

    # Compute the determinant to remove the elements that are
    # causing singularity.
    a_det = np.linalg.det(a)

    # Set the change the values of the places where a singular matrix is received.
    # In the case of matrix a we change the values to an identity matrix. b is set
    # to zeros.
    a[a_det == 0] = np.array([[1, 0], [0, 1]])
    b[a_det == 0] = np.array([0, 0])

    # Solve the equation
    x = np.linalg.solve(a, b)
    u = x[:, :, 0]
    v = x[:, :, 1]

    return u, v

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    w = np.array([1, 4, 6, 4, 1]) / 16
    filtered_image = cv2.sepFilter2D(image, cv2.CV_64F, w, w)

    return filtered_image[::2, ::2]


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    gaussian_pyramid = [np.copy(image)]

    for _ in range(levels-1):
        new_level = reduce_image(gaussian_pyramid[-1])
        gaussian_pyramid.append(new_level)

    return gaussian_pyramid


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    x_size = 0
    y_size = img_list[0].shape[0]
    for img in img_list:
        x_size += img.shape[1]

    dst = np.zeros((y_size, x_size))
    x_position = 0
    for img in img_list:
        dst[0:img.shape[0], x_position:x_position+img.shape[1]] = img
        x_position = x_position+img.shape[1]

    return dst


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    shape = image.shape
    spaced = np.zeros((2*shape[0], 2*shape[1]))
    spaced[::2,::2] = image

    w = np.array([1, 4, 6, 4, 1]) / 8
    dst = cv2.sepFilter2D(spaced, cv2.CV_64F, w, w)

    return dst

def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    # Compute the difference at each l.
    l_pyr = [g_pyr[i-1] - expand_image(g_pyr[i]) for i in range(1,len(g_pyr))]

    # Add the smallest gaussian at the very end
    l_pyr.append(g_pyr[-1])

    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    # Based on the code snippet provided in the description of PS4
    rows, columns = image.shape
    X, Y = np.meshgrid(range(columns), range(rows))

    X = X + U
    Y = Y + V

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)


    result = np.zeros(image.shape)
    cv2.remap(image, X, Y, interpolation, dst=result, borderMode=border_mode)

    return result


def scale_u_and_v(u, v, level, pyr):
    for _ in range(level):
        u = 2 * expand_image(u)
        v = 2 * expand_image(v)

    print('scale:', u.shape)
    shape = pyr[0].shape
    return u[:shape[0], :shape[1]], v[:shape[0], :shape[1]]



def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    pyr_a = gaussian_pyramid(img_a, levels)
    pyr_b = gaussian_pyramid(img_b, levels)

    # Initialize u and v with the sizes of level K+1. Note that this
    # value is not part of the gaussian pyramid. So, we compute its
    # size by removing the pixels as it is removed in the REDUCE
    # operation.
    u = np.zeros(pyr_a[-1].shape)[::2, ::2]
    v = np.zeros(pyr_a[-1].shape)[::2, ::2]

    for level in range(levels-1, -1, -1):

        # Expand the u and v array.
        u_p = 2 * expand_image(u)
        v_p = 2 * expand_image(v)

        # Remove excessive edges if any after the expansion process.
        shape = pyr_a[level].shape
        u_p, v_p = u_p[:shape[0], :shape[1]], v_p[:shape[0], :shape[1]]

        # Compute the warped version of image b.
        warped = warp(pyr_b[level], u_p, v_p, interpolation, border_mode)
        u_delta, v_delta = optic_flow_lk(pyr_a[level], warped, k_size, k_type, sigma)

        # Add the resulting delta.
        u = u_p + u_delta
        v = v_p + v_delta

    return u, v

###################################################################################
# import os

# # I/O directories
# input_dir = "input_images"
# output_dir = "./"

# def quiver(u, v, scale, stride, color=(0, 255, 0)):

#     img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

#     for y in range(0, v.shape[0], stride):

#         for x in range(0, u.shape[1], stride):

#             cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
#                                        y + int(v[y, x] * scale)), color, 1)
#             cv2.circle(img_out, (x + int(u[y, x] * scale),
#                                  y + int(v[y, x] * scale)), 1, color, 1)
#     return img_out


# if __name__ == '__main__':
#     shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
#                                       'Shift0.png'), 0) / 255.
#     shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
#                                         'ShiftR10.png'), 0) / 255.
#     shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
#                                         'ShiftR20.png'), 0) / 255.
#     shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
#                                         'ShiftR40.png'), 0) / 255.
#     # =======
#     # PART B1
#     # =======

#     levels = 5
#     k_size = 65
#     k_type = 'gaussian'
#     sigma = 30
#     interpolation = cv2.INTER_CUBIC
#     border_mode = cv2.BORDER_REFLECT101
#     u, v = hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type, sigma, interpolation, border_mode)

#     u_v = quiver(u, v, scale=1.5, stride=10)
#     cv2.imwrite(os.path.join(output_dir, "last.png"), u_v)