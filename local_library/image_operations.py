import numpy as np
from scipy.optimize import linear_sum_assignment
from pyzbar.pyzbar import decode
from skimage.filters import threshold_local


def convertRGBtoYCbCr(image_RGB):
    """
    Convert an RGB image or vector to the YCbCr color space.

    Parameters:
        image_RGB (np.ndarray): Input RGB image or vector with values in the range [0, 1] or [0, 255].
                                The input can be a 3D array representing an image (height x width x 3)
                                or a 2D array representing a vector (n x 3).

    Returns:
        np.ndarray: YCbCr image or vector with values in the range [0, 1]. The output will have the
                    same shape as the input.
    """

    # check if the RGB image is in range 0-255 -> divide by 255 -> float image
    if np.max(image_RGB) > 1:
        image_RGB = image_RGB / 255

    # separate the color values of each channel (Red, Green, Blue)
    if len(image_RGB.shape) == 2:  # vector
        r, g, b = image_RGB[:, 0], image_RGB[:, 1], image_RGB[:, 2]
    elif len(image_RGB.shape) == 3:  # image
        r, g, b = image_RGB[:, :, 0], image_RGB[:, :, 1], image_RGB[:, :, 2]

    # calculate Y, Cb and Cr
    kr, kg, kb = 0.299, 0.587, 0.114
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))

    # combine the color values of each channel back
    output_img_ybr = np.stack([y, cb, cr], axis=-1)
    if len(image_RGB.shape) == 2:  # vector
        return np.asarray(output_img_ybr)
    elif len(image_RGB.shape) == 3:  # image
        return output_img_ybr


def convertYCbCrtoRGB(image_YCbCr):
    """
    Convert a YCbCr image or vector to the RGB color space.

    Parameters:
        image_YCbCr (np.ndarray): Input YCbCr image or vector with values in the range [0, 1].
                                  The input can be a 3D array representing an image (height x width x 3)
                                  or a 2D array representing a vector (n x 3).

    Returns:
        np.ndarray: RGB image or vector with values in the range [0, 1]. The output will have the
                    same shape as the input.
    """

    # separate the color values of each channel (Y, Cb, Cr)
    if len(image_YCbCr.shape) == 2:  # vector
        y, cb, cr = image_YCbCr[:, 0], image_YCbCr[:, 1], image_YCbCr[:, 2]
    elif len(image_YCbCr.shape) == 3:  # image
        y, cb, cr = image_YCbCr[:, :, 0], image_YCbCr[:, :, 1], image_YCbCr[:, :, 2]
    elif len(image_YCbCr.shape) == 1:  # pixel
        y, cb, cr = image_YCbCr[0], image_YCbCr[1], image_YCbCr[2]

    # calculate R, G and B
    kr, kg, kb = 0.299, 0.587, 0.114
    r = y + (2 - 2 * kr) * cr
    g = y - ((kb / kg) * (2 - 2 * kb) * cb) - ((kr / kg) * (2 - 2 * kr) * cb)
    b = y + (2 - 2 * kb) * cb

    # combine the color values of each channel back
    output_img_rgb = np.stack([r, g, b], axis=-1)
    output_img_rgb = np.clip(output_img_rgb, 0, 1)

    if len(image_YCbCr.shape) == 2:  # vector
        return np.asarray(output_img_rgb)
    elif len(image_YCbCr.shape) == 3:  # image
        return output_img_rgb
    elif len(image_YCbCr.shape) == 1:  # pixel
        return output_img_rgb


def resizeImage(image_input, new_height, new_width):
    """
    Resize an image to the specified width and height.

    Parameters:
        image_input (np.ndarray): Input image to be resized.
        new_height (int): Height of the resized image (number of rows).
        new_width (int): Width of the resized image (number of columns).

    Returns:
        np.ndarray: Resized image with the specified dimensions.
    """

    # get shape of input image
    height_input, width_input = image_input.shape[:2]

    # Calculate scaling factors for resizing
    width_scale = float(new_width) / width_input
    height_scale = float(new_height) / height_input

    # get the position of each pixel of output image
    grid_y, grid_x = np.mgrid[:new_height, :new_width]

    # calculate the position where the output image will get the color values from the input image
    orig_y = (grid_y / height_scale).astype(int)
    orig_x = (grid_x / width_scale).astype(int)

    # Ensure the positions are within the bounds of the input image
    orig_y = np.clip(orig_y, 0, height_input - 1)
    orig_x = np.clip(orig_x, 0, width_input - 1)

    # create the output image from input image at the calculated position
    return image_input[orig_y, orig_x]


def calculateMultiGaussianDist(input_image, mean, covariance):
    """
    Calculate the Probability Density Function (PDF) of a Multivariate Gaussian Distribution for each pixel.

    Parameters:
        input_image (np.ndarray): Input color image for density calculation.
        mean (np.ndarray): Mean vector of the Gaussian distribution (shape: [n]).
        covariance (np.ndarray): Covariance matrix of the Gaussian distribution (shape: [n, n]).

    Returns:
        np.ndarray: Density image (likelihood) with the same shape as the input image (height, width).
    """

    # convert image to vector
    input_vector = input_image.reshape(-1, input_image.shape[-1])
    # get the dimension of data ( ex. 3 colors = 3d )
    dimension = mean.shape[0]
    # calculate determinant and inverse matrix of covariance matrix
    det_covariance = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)
    # calculate normalization factor in PDF formula
    normalization_factor = 1 / np.sqrt(((2 * np.pi) ** dimension) * det_covariance)
    # calculate exponential in PDF formula
    mean = np.ones_like(input_vector) * mean  # create mean with shape like input vector
    exponential = np.sum(
        (np.dot((input_vector - mean), inv_covariance)) * (input_vector - mean),
        axis=1,
    )
    # calculate density from normalization factor and exponential
    density_vector = normalization_factor * np.exp(-0.5 * exponential)
    # reshape from vector to image like input image
    density_image = density_vector.reshape(input_image.shape[:2])
    return density_image


def warpPerspectiveTransformation(
    image, source_corner_positions, output_height, output_width
):
    """
    Perform a perspective transformation on the given image.

    Parameters:
        image (np.ndarray): Input image to be transformed.
        source_corner_positions (np.ndarray): Source corner positions of the object to be transformed (shape: [4, 2]).
        output_height (int): Height of the output image after transformation.
        output_width (int): Width of the output image after transformation.

    Returns:
        np.ndarray: Transformed image with the perspective applied.
    """

    # Define the output positions as a rectangle
    output_positions = np.array(
        [[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]]
    )

    # Extract x and y coordinates of positions
    source_x, source_y = source_corner_positions[:, 0], source_corner_positions[:, 1]
    target_x, target_y = output_positions[:, 0], output_positions[:, 1]

    # Create the transform matrix for linear transformation
    transform_matrix = (
        np.array(
            [
                [
                    source_x,
                    source_y,
                    np.ones_like(source_x),
                    np.zeros_like(source_x),
                    np.zeros_like(source_x),
                    np.zeros_like(source_x),
                    -source_x * target_x,
                    -source_y * target_x,
                ],
                [
                    np.zeros_like(source_x),
                    np.zeros_like(source_x),
                    np.zeros_like(source_x),
                    source_x,
                    source_y,
                    np.ones_like(source_x),
                    -source_x * target_y,
                    -source_y * target_y,
                ],
            ]
        )
        .transpose(2, 0, 1)
        .reshape(-1, 8)
    )

    # Define the target positions as a flat array
    target_positions = output_positions.flatten()

    # Solve for the transformation matrix using least squares method
    transform_elements, residuals, rank, s = np.linalg.lstsq(
        transform_matrix, target_positions, rcond=None
    )
    transformation_matrix = np.append(transform_elements, 1).reshape(3, 3)

    # Generate all positions in the output image
    all_positions = np.stack(np.mgrid[:output_height, :output_width][::-1], axis=2)

    # Convert positions to homogeneous coordinates
    vector_positions = np.vstack(
        (
            all_positions.reshape(-1, 2).T,
            np.ones((1, output_width * output_height), dtype=int),
        )
    )

    # Apply the inverse perspective transformation matrix
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)
    transformed_positions = np.dot(inv_transformation_matrix, vector_positions)
    transformed_positions = (
        (transformed_positions / transformed_positions[-1]).astype(int)[:-1].T
    )

    # # Clip positions to stay within image bounds
    # transformed_positions = np.clip(
    #     transformed_positions.astype(int), 0, np.asarray(image.shape[:2]) - 1
    # )

    # Map pixels from input image to output image based on transformed positions
    output_image = image[
        transformed_positions[:, 1], transformed_positions[:, 0]
    ].reshape(output_height, output_width, 3)

    return output_image


def calculateMahalanobis(x, mean, cov):
    """
    Calculate the Mahalanobis distance between data points and a multivariate Gaussian distribution.

    Parameters:
        x (np.ndarray): Data points with shape (h, w, n) or (h*w, n), where (h, w) represents image dimensions
                        and n is the number of features (e.g., color channels). For image data, the last dimension
                        should match the dimensionality of the mean and covariance matrix.
        mean (np.ndarray): Mean vector of the multivariate Gaussian distribution with shape (n,).
        cov (np.ndarray): Covariance matrix of the multivariate Gaussian distribution with shape (n, n).

    Returns:
        np.ndarray: Mahalanobis distances with shape (h, w, 1) or (h*w, 1), representing the distance of each
                    data point from the mean, accounting for the covariance.
    """

    # Calculate the difference between each data point and the mean
    diff = x - mean
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    # Calculate the Mahalanobis distance
    mahalanobis_dist = np.sqrt((np.dot(diff, inv_cov) * diff).sum(-1))

    return mahalanobis_dist


def calculateBhattacharyya(mean1, cov1, mean2, cov2):
    """
    Calculate the Bhattacharyya distance between two multivariate Gaussian distributions.

    Parameters:
        mean1 (np.ndarray): Mean vector of the first Gaussian distribution.
        cov1 (np.ndarray): Covariance matrix of the first Gaussian distribution.
        mean2 (np.ndarray): Mean vector of the second Gaussian distribution.
        cov2 (np.ndarray): Covariance matrix of the second Gaussian distribution.

    Returns:
       np.ndarray: Matrix of Bhattacharyya distances with shape
                    (number of Gaussian models in the first GMM, number of Gaussian models in the second GMM),
                    where each element (i, j) represents the distance between the i-th Gaussian of the first GMM
                    and the j-th Gaussian of the second GMM.
    """

    # Calculate the mean difference between each pair of Gaussian models
    mean_diff = mean1 - mean2
    # Calculate the average covariance matrix for each pair of Gaussian models
    cov = (cov1 + cov2) / 2
    # Calculate the Mahalanobis term for each pair of Gaussian models
    mahalanobis_term = (np.dot(mean_diff, cov) * mean_diff).sum(-1)
    # Calculate the log term for each pair of Gaussian models
    log_term = np.linalg.det(cov) / (np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
    # Calculate the Bhattacharyya distance for each pair of Gaussian models
    bhattacharyya_dist = (0.125 * mahalanobis_term) + (0.5 * np.log(log_term))

    return bhattacharyya_dist


def pairGaussianModels(gmm_mean_1, gmm_cov_1, gmm_mean_2, gmm_cov_2):
    """
    Pair Gaussian models between two GMMs using Bhattacharyya distance.

    Parameters:
        gmm_mean_1 (np.ndarray): Array of mean vectors for the first GMM.
        gmm_cov_1 (np.ndarray): Array of covariance matrices for the first GMM.
        gmm_mean_2 (np.ndarray): Array of mean vectors for the second GMM.
        gmm_cov_2 (np.ndarray): Array of covariance matrices for the second GMM.

    Returns:
        tuple: Indices of the paired Gaussian models from the two GMMs.
    """

    # Number of Gaussian models in each GMM
    n_gmm = [gmm_mean_1.shape[0], gmm_mean_2.shape[0]]
    # Initialize distance matrix
    distance = np.zeros(n_gmm)
    # Calculate Bhattacharyya distance for each pair of Gaussian models
    for index_1 in range(n_gmm[0]):
        for index_2 in range(n_gmm[1]):
            distance[index_1][index_2] = calculateBhattacharyya(
                gmm_mean_1[index_1],
                gmm_cov_1[index_1],
                gmm_mean_2[index_2],
                gmm_cov_2[index_2],
            )
    # Use linear sum assignment to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance)
    return row_ind, col_ind


def calculateMeanAndCovariance(vector, weight=None):
    """
    Calculate the mean and covariance matrix of a multi-dimensional array.

    Parameters:
        vector (np.ndarray): Input array with shape (N, M), where N is the number of samples
                            and M is the number of features.
        weight (np.ndarray): Weights for the samples, with shape (N,).

    Returns:
        tuple: A tuple containing:
            - Mean vector (np.ndarray): Mean of each feature.
            - Covariance matrix (np.ndarray): Covariance matrix of the features.
    """
    # Reshape the array to be 2D where the second dimension represents features
    if len(vector.shape) > 2:
        vector = vector.reshape(-1, vector.shape[-1])

    # Flatten the weight array if it has more than one dimension
    if np.any(weight != None) and len(weight.shape) > 1:
        weight = weight.reshape(-1)

    # Calculate the mean vector, considering weights if provided
    mean_vector = np.average(vector, axis=0, weights=weight)

    # Calculate the covariance matrix, considering weights if provided
    covariance_matrix = np.cov(vector.T, aweights=weight)

    return [mean_vector, covariance_matrix]


def calculateBayes(foreground_prob, background_prob):
    """
    Calculate the Bayesian probability for each element in the provided arrays of foreground and background probabilities.

    Formula:
        BayesProbability = 1 / (1 + (BackgroundProbability / ForegroundProbability))

    Parameters:
        foreground_prob (np.ndarray): The array of probabilities for the foreground.
        background_prob (np.ndarray): The array of probabilities for the background.

    Returns:
        np.ndarray: The array of calculated Bayesian probabilities.
    """

    # Compute the Bayesian probability for each element in the arrays using the formula

    BayesProbability = np.divide(
        1,
        (
            1
            + np.divide(
                background_prob,
                foreground_prob,
                out=np.zeros_like(background_prob),
                where=foreground_prob != 0,
            )
        ),
        out=np.zeros_like(background_prob),
        where=foreground_prob != 0,
    )

    return BayesProbability


def displayEllipseOnImage(
    resized_rgb_image,
    status,
    mean_positions,
    covariance_matrices,
    row_count,
    column_count,
):
    """
    Overlay ellipses onto a resized RGB image based on means and covariances for cells in a tray.

    Parameters:
        resized_rgb_image (np.array): The RGB image to overlay ellipses onto.
        status (np.array): A 2D array indicating whether a cell contains a valid object (1) or not (0).
        mean_positions (np.array): A 2D array of mean positions (centroids) for each cell.
        covariance_matrices (np.array): A 2D array of covariance matrices for each cell.
        row_count (int): Number of rows in the tray grid.
        column_count (int): Number of columns in the tray grid.

    Returns:
        np.array: The image with ellipses overlaid onto it.
    """

    # Copy the image for ellipse overlay
    ellipse_image = resized_rgb_image.copy()

    # Get the shape of the image and calculate the shape of each cell based on row and column counts
    image_height, image_width = ellipse_image.shape[:2]
    cell_shape = np.array([image_height, image_width]) // np.array(
        [row_count, column_count]
    )

    # Create meshgrid of row and column indices based on tray dimensions
    tray_row_indices, tray_col_indices = np.meshgrid(
        range(row_count), range(column_count)
    )

    # Stack row and column indices to form a 2D array representing cell indices in the tray
    tray_cell_indices = np.stack([tray_row_indices.T, tray_col_indices.T], axis=-1)

    # Calculate the centroid of each cell in the tray
    cell_centroids = (cell_shape // 2) + tray_cell_indices * cell_shape

    # Iterate over each cell in the tray
    for row_index in range(row_count):
        for col_index in range(column_count):

            # If the cell has a valid status, proceed with ellipse calculation
            if status[row_index, col_index] == 1:

                # Calculate the relative top-left and bottom-right positions of the pixel grid
                top_left_relative = -cell_centroids[row_index, col_index]
                bottom_right_relative = (
                    np.array([image_height, image_width])
                    - cell_centroids[row_index, col_index]
                )

                # Generate position vectors for each pixel within the cell
                pixel_row_indices, pixel_col_indices = np.meshgrid(
                    range(top_left_relative[0], bottom_right_relative[0]),
                    range(top_left_relative[1], bottom_right_relative[1]),
                )
                pixel_positions = np.stack(
                    [pixel_row_indices.T, pixel_col_indices.T], axis=-1
                )

                # Calculate eigenvalues and eigenvectors for the covariance matrix of the current cell
                eigenvalues, eigenvectors = np.linalg.eig(
                    covariance_matrices[row_index, col_index]
                )

                # Calculate the semi-major and semi-minor axes using the eigenvalues
                # semi_axes_squared = 4.605 * eigenvalues  # For 90% confidence level
                semi_axes_squared = 5.991 * eigenvalues  # For 95% confidence level
                # semi_axes_squared = 9.210 * eigenvalues  # For 99% confidence level

                # Calculate the orientation of the ellipse
                orientation_angle = -np.arctan(eigenvectors[1, 0] / eigenvectors[0, 0])

                # Calculate distances of pixels from the mean positions in the x and y directions
                euclidean_dist_x = (
                    pixel_positions[:, :, 0] - mean_positions[row_index, col_index, 0]
                )
                euclidean_dist_y = (
                    pixel_positions[:, :, 1] - mean_positions[row_index, col_index, 1]
                )

                # Rotate the pixel coordinates based on the ellipse's orientation
                x_rot = euclidean_dist_x * np.cos(
                    orientation_angle
                ) - euclidean_dist_y * np.sin(orientation_angle)
                y_rot = euclidean_dist_x * np.sin(
                    orientation_angle
                ) + euclidean_dist_y * np.cos(orientation_angle)

                # Ellipse equation in its standard form
                ellipse_equation = (
                    (x_rot**2 / semi_axes_squared[0])
                    + (y_rot**2 / semi_axes_squared[1])
                    - 1
                )

                # Calculate the area of the ellipse
                ellipse_area = np.pi * np.sqrt(semi_axes_squared).prod()

                # Add the ellipse to the image by coloring pixels that satisfy the ellipse equation
                # ellipse_image += np.where( (ellipse_equation < 10 / np.sqrt(ellipse_area)) & (ellipse_equation > 0), 1, 0)[:, :, None]
                ellipse_image[
                    np.where(
                        (ellipse_equation < 10 / np.sqrt(ellipse_area))
                        & (ellipse_equation > 0)
                    )
                ] = 1

    return ellipse_image


def displayEllipseOnPlantingTray(
    resized_rgb_image,
    mean_positions,
    covariance_matrices,
):
    """
    Overlay ellipses onto a resized RGB image for planting tray based on means and covariances for cells in a tray.

    Parameters:
        resized_rgb_image (np.array): The RGB image to overlay ellipses onto.
        mean_positions (np.array): A 2D array of mean positions (centroids) for each cell.
        covariance_matrices (np.array): A 2D array of covariance matrices for each cell.

    Returns:
        np.array: The image with ellipses overlaid onto it.
    """

    # Copy the image for ellipse overlay
    ellipse_image = resized_rgb_image.copy()

    # Generate position vectors for each pixel
    pixel_row_indices, pixel_col_indices = np.meshgrid(
        range(0, ellipse_image.shape[0]),
        range(0, ellipse_image.shape[1]),
    )
    pixel_positions = np.stack([pixel_row_indices.T, pixel_col_indices.T], axis=-1)

    # Iterate over each cell in the tray
    for index in range(len(mean_positions)):

        # Calculate eigenvalues and eigenvectors for the covariance matrix of the current cell
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrices[index])

        # Calculate the semi-major and semi-minor axes using the eigenvalues
        semi_axes_squared = 4.605 * eigenvalues  # For 90% confidence level
        # semi_axes_squared = 5.991 * eigenvalues  # For 95% confidence level
        # semi_axes_squared = 9.210 * eigenvalues  # For 99% confidence level

        # Calculate the orientation of the ellipse
        orientation_angle = -np.arctan(eigenvectors[1, 0] / eigenvectors[0, 0])

        # Calculate distances of pixels from the mean positions in the x and y directions
        euclidean_dist_x = pixel_positions[:, :, 0] - mean_positions[index, 0]
        euclidean_dist_y = pixel_positions[:, :, 1] - mean_positions[index, 1]

        # Rotate the pixel coordinates based on the ellipse's orientation
        x_rot = euclidean_dist_x * np.cos(
            orientation_angle
        ) - euclidean_dist_y * np.sin(orientation_angle)
        y_rot = euclidean_dist_x * np.sin(
            orientation_angle
        ) + euclidean_dist_y * np.cos(orientation_angle)

        # Ellipse equation in its standard form
        ellipse_equation = (
            (x_rot**2 / semi_axes_squared[0]) + (y_rot**2 / semi_axes_squared[1]) - 1
        )

        # Calculate the area of the ellipse
        ellipse_area = np.pi * np.sqrt(semi_axes_squared).prod()

        # Add the ellipse to the image by coloring pixels that satisfy the ellipse equation
        # ellipse_image += np.where( (ellipse_equation < 10 / np.sqrt(ellipse_area)) & (ellipse_equation > 0), 1, 0)[:, :, None]
        ellipse_image[
            np.where(
                (ellipse_equation < 10 / np.sqrt(ellipse_area)) & (ellipse_equation > 0)
            )
        ] = 1

    return ellipse_image


def createPrior(
    cov_ratio=20, image_shape=np.array([160, 320, 3]), n_row=8, n_column=16
):
    """
    Create a prior probability map based on Gaussian distribution for each cell in the image.

    Parameters:
        cov_ratio (int): Ratio used to determine the covariance matrix scale.
        image_shape (np.ndarray): Shape of the image (height, width, channels). Only the first two dimensions are used.
        n_row (int): Number of rows of cells in the image grid.
        n_column (int): Number of columns of cells in the image grid.

    Returns:
        np.ndarray: The prior probability map for the entire image.
    """

    # If the image shape includes channels, use only the height and width
    if len(image_shape) == 3:
        image_shape = image_shape[:2]

    # Calculate the shape of each cell in the grid
    cell_shape = np.ceil(image_shape / np.array([n_row, n_column])).astype(int)
    cell_shape_float = image_shape / np.array([n_row, n_column])

    # Initialize arrays for prior probabilities
    prior_each_cell = np.zeros(cell_shape)
    prior_all_cell = np.zeros(image_shape)

    # Define the centroid and covariance for the Gaussian distribution in each cell
    centroid_each_cell = np.array(cell_shape // 2)
    covariance_each_cell = (
        np.array([[cell_shape[0], 0], [0, cell_shape[1]]]) * cov_ratio
    )

    # Create a grid of positions for each cell
    col_indices, row_indices = np.meshgrid(range(cell_shape[1]), range(cell_shape[0]))
    position_each_cell = np.stack([row_indices, col_indices], axis=-1)

    # Calculate the Gaussian distribution for each cell
    prior_each_cell = calculateMultiGaussianDist(
        position_each_cell, centroid_each_cell, covariance_each_cell
    )

    # Aggregate the Gaussian distributions across all cells in the image
    for row_cell in range(n_row):
        for col_cell in range(n_column):

            kernel_start = np.round(
                cell_shape_float * np.array([row_cell, col_cell])
            ).astype(int)
            kernel_end = kernel_start + cell_shape.astype(int)

            prior_all_cell[
                kernel_start[0] : kernel_end[0], kernel_start[1] : kernel_end[1]
            ] += prior_each_cell

    return prior_all_cell


def calculateEllipseArea(covariance_matrix):
    """
    Calculate the area of an ellipse given its covariance matrix.

    The area of the ellipse is determined by the eigenvalues of the covariance matrix.
    The eigenvalues represent the squared lengths of the semi-major and semi-minor axes of the ellipse.
    The area is calculated as π multiplied by the square root of the eigenvalues, representing the semi-axes' lengths,
    and then taking the product of those values to get the total area.

    Parameters:
    covariance_matrix (numpy.ndarray): A 2x2 covariance matrix representing the ellipse shape.

    Returns:
    float: The calculated area of the ellipse.
    """

    # Calculate the eigenvalues (lengths of semi-major and semi-minor axes) and eigenvectors (directions) of the ellipse
    eigenvalues = np.linalg.eigvals(covariance_matrix)

    # Calculate squared semi-axes lengths for a confidence interval
    semi_axes_squared = 4.605 * eigenvalues  # For 90% confidence level
    # semi_axes_squared = 5.991 * eigenvalues  # For 95% confidence level
    # semi_axes_squared = 9.210 * eigenvalues  # For 99% confidence level

    # Calculate the area of the ellipse (π * product of semi-major and semi-minor axes)
    ellipse_area = np.pi * np.sqrt(semi_axes_squared).prod(-1)

    return ellipse_area


def checkOutlier(status, data):
    """
    Identify outliers in the data based on the interquartile range (IQR) method.

    Parameters:
        status (np.ndarray): Boolean or binary array indicating the cells to check for outliers (e.g., whether a cell
                             contains a seedling).
        data (np.ndarray): Array of numerical values (e.g., seedling sizes or color distances) for which outliers are
                           to be detected. Only values corresponding to True status will be considered.

    Returns:
        np.ndarray: Array of coordinates (indices) where outliers are detected in the input data. The coordinates
                    are stacked in the format (row, column) for 2D data.
    """

    # Compute the first and third quartiles (Q1 and Q3)
    q1, q3 = np.percentile(data[status], [25, 75])
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    # Calculate the lower bound (Q1 - 1.5 * IQR) and upper bound (Q3 + 1.5 * IQR)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Identify the outliers as values less than the lower bound or greater than the upper bound
    outliers = np.logical_or(data < lower_bound, data > upper_bound)

    # Stack the coordinates of outliers (where status is True) in (row, column) format
    return np.stack(np.where(status & outliers), axis=-1)


def calculateEuclidean(mean1, mean2):
    """
    Calculate the Euclidean distance between two points or vectors.

    Parameters:
        mean1 (np.ndarray): Vector of the first point.
        mean2 (np.ndarray): Vector of the second point.

    Returns:
       np.ndarray: Euclidean distance between the points.
    """

    # Calculate the difference between the vectors
    mean_diff = mean1 - mean2
    # Calculate the Euclidean distance
    euclidean_dist = np.sqrt(np.sum(mean_diff**2, axis=-1))

    return euclidean_dist


def decodeQR(RGB_image):
    """
    Decodes a QR code from an RGB image using local thresholding and sigmoid transformation.

    Parameters:
    RGB_image (numpy.ndarray): The input image in RGB format.

    Returns:
    list: Decoded QR code information, if any is detected.
    """
    # Convert RGB image to grayscale using YCbCr luminance calculation
    ycbcr_constant = np.array([0.299, 0.587, 0.114])
    y_image = (RGB_image * ycbcr_constant).sum(axis=-1)

    # Normalize and convert to integer scale if needed
    y_image = (y_image * 255).astype(int) if y_image.max() <= 1 else y_image.astype(int)

    # Apply a local threshold to the grayscale image
    local_thresh = threshold_local(y_image, block_size=65, offset=10)

    # Perform sigmoid transformation for enhanced contrast
    sigmoid_local = (
        1 / (1 + np.exp(-10 * ((y_image / 255) - (local_thresh / 255)))) * 255
    ).astype(int)

    # Decode the QR code from the processed image
    decode_sigmoid_local = decode(sigmoid_local)

    return decode_sigmoid_local


def createSeedlingEllipseMask(
    image_shape, status, mean_positions, covariance_matrices, row_count, column_count
):
    """
    Overlay ellipses onto a resized RGB image based on means and covariances for cells in a tray.

    Parameters:
        resized_rgb_image (np.array): The RGB image to overlay ellipses onto.
        status (np.array): A 2D array indicating whether a cell contains a valid object (1) or not (0).
        mean_positions (np.array): A 2D array of mean positions (centroids) for each cell.
        covariance_matrices (np.array): A 2D array of covariance matrices for each cell.
        row_count (int): Number of rows in the tray grid.
        column_count (int): Number of columns in the tray grid.

    Returns:
        np.array: The image with ellipses overlaid onto it.
    """

    # Copy the image for ellipse overlay
    mask = np.zeros(image_shape)

    # Get the shape of the image and calculate the shape of each cell based on row and column counts
    cell_shape = image_shape // np.array([row_count, column_count])

    # Create meshgrid of row and column indices based on tray dimensions
    tray_row_indices, tray_col_indices = np.meshgrid(
        range(row_count), range(column_count)
    )

    # Stack row and column indices to form a 2D array representing cell indices in the tray
    tray_cell_indices = np.stack([tray_row_indices.T, tray_col_indices.T], axis=-1)

    # Calculate the centroid of each cell in the tray
    cell_centroids = (cell_shape // 2) + tray_cell_indices * cell_shape

    # Iterate over each cell in the tray
    for row_index in range(row_count):
        for col_index in range(column_count):

            # If the cell has a valid status, proceed with ellipse calculation
            if status[row_index, col_index] == 1:

                # Calculate the relative top-left and bottom-right positions of the pixel grid
                top_left_relative = -cell_centroids[row_index, col_index]
                bottom_right_relative = (
                    image_shape - cell_centroids[row_index, col_index]
                )

                # Generate position vectors for each pixel within the cell
                pixel_row_indices, pixel_col_indices = np.meshgrid(
                    range(top_left_relative[0], bottom_right_relative[0]),
                    range(top_left_relative[1], bottom_right_relative[1]),
                )
                pixel_positions = np.stack(
                    [pixel_row_indices.T, pixel_col_indices.T], axis=-1
                )

                # Calculate eigenvalues and eigenvectors for the covariance matrix of the current cell
                eigenvalues, eigenvectors = np.linalg.eig(
                    covariance_matrices[row_index, col_index]
                )

                # Calculate the semi-major and semi-minor axes using the eigenvalues
                # semi_axes_squared = 4.605 * eigenvalues  # For 90% confidence level
                # semi_axes_squared = 5.991 * eigenvalues  # For 95% confidence level
                semi_axes_squared = 9.210 * eigenvalues  # For 99% confidence level

                # Calculate the orientation of the ellipse
                orientation_angle = -np.arctan(eigenvectors[1, 0] / eigenvectors[0, 0])

                # Calculate distances of pixels from the mean positions in the x and y directions
                euclidean_dist_x = (
                    pixel_positions[:, :, 0] - mean_positions[row_index, col_index, 0]
                )
                euclidean_dist_y = (
                    pixel_positions[:, :, 1] - mean_positions[row_index, col_index, 1]
                )

                # Rotate the pixel coordinates based on the ellipse's orientation
                x_rot = euclidean_dist_x * np.cos(
                    orientation_angle
                ) - euclidean_dist_y * np.sin(orientation_angle)
                y_rot = euclidean_dist_x * np.sin(
                    orientation_angle
                ) + euclidean_dist_y * np.cos(orientation_angle)

                # Ellipse equation in its standard form
                ellipse_equation = (
                    (x_rot**2 / semi_axes_squared[0])
                    + (y_rot**2 / semi_axes_squared[1])
                    - 1
                )
                # Calculate the area of the ellipse
                ellipse_area = np.pi * np.sqrt(semi_axes_squared).prod()

                # Add the ellipse to the image by coloring pixels that satisfy the ellipse equation
                # ellipse_image += np.where( (ellipse_equation < 10 / np.sqrt(ellipse_area)) & (ellipse_equation > 0), 1, 0)[:, :, None]
                mask[np.where((ellipse_equation < 10 / np.sqrt(ellipse_area)))] = 1

    return mask
