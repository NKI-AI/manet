"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np

def roi_generation(
        coordinates, image, spacing, mask_for_visualization, file_name, out_dir, patch_size, bbox_size, pixel_size, count):
    g = None
    compact = None
    fontSize = 10
    patch_size_for_visualization = 8
    len = coordinates.size[0]
    num_pixels_vector = []

    for i in range(coordinates.size[0]):
        X_ph = coordinates[i, 0]
        Y_ph = coordinates[i, 1]
        X_pix = np.round(X_ph * [pixel_size / spacing])
        Y_pix = np.round(Y_ph * [pixel_size / spacing])

        if Y_pix - patch_size < 0 | X_pix - patch_size < 0:
            count += 1
            if Y_pix - patch_size < 0:
                padding = np.ceil(patch_size - Y_pix)
                patch_size = patch_size - padding - 1
                patch_size_for_visualization = patch_size


            if (X_pix - patch_size < 0)
                padding =  np.ceil(patch_size - X_pix)
                patch_size = patch_size - padding - 1
                patch_size_for_visualization = patch_size

        cropped = image[Y_pix - patch_size: Y_pix + patch_size, X_pix - patch_size: X_pix + patch_size,:]

        max_value_bb = image(Y_pix, X_pix);
        value_to_debugY = Y_pix - bbox_size;
        value_to_debugX = X_pix - bbox_size;
        value_to_debugY2 = Y_pix + bbox_size;
        value_to_debugX2 = X_pix + bbox_size;

        bbox = image(Y_pix - bbox_size: Y_pix + bbox_size, X_pix - bbox_size: X_pix + bbox_size,:);
        bbox_leftup_X = X_pix - bbox_size;
        bbox_leftup_Y = Y_pix - bbox_size;
        bbox_Rup_X = X_pix + bbox_size;
        bbox_Rup_Y = Y_pix + bbox_size;

        center_points_bb = find_new_center_bbox(image, max_value_bb, X_pix, Y_pix, bbox_size);
        % center_points_bb = find_new_center_circle(image, pline_x, pline_y, max_value, X_pix, Y_pix);

        return max_numberOFpixels, min_numberOFpixels, mean_numberOFpixels, mask_for_visulatization