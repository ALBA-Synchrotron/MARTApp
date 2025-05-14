#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2022 ALBA Synchrotron
#
# Authors: Joaquin Gomez Sanchez, A. Estela Herguedas Alonso
#
# This file is part of Mistral beamline software.
# (see https://www.albasynchrotron.es/en/beamlines/bl09-mistral)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import warnings

warnings.filterwarnings("error")

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from joblib import Parallel, delayed
from matplotlib.widgets import Slider
from PyQt5.QtWidgets import QInputDialog, QMessageBox
from pystackreg import StackReg
from skimage.registration import phase_cross_correlation
from artis_tomo.image.filter import maskRaisedCosineRadial
from artis_tomo.programs.tiltalign import tilt_align

from sdm.mistral.image import align, mv_projection, mv_projection_subpixel
from sdm.mistral.scripts.parser import base_parser
from sdm.mistral.util import select_roi


WORKFLOW = (
    "[exclude angles] >> [crop] >> [create mask] >> 2D alignment polarizations"
    " >> crop >> compensation >> -ln(stacks) >> pos_stack +- neg_stack"
    " >> [tilt alignment]"
)


def app_parser():
    """
    Defines parser arguments for the magnetism_xmcd pipeline.
    """
    description = (
        "This pipeline computes the absorbtion and the signal of two given "
        "polarization stacks previously computed using the magnetism pipeline. "
        "It offers different 2D alignment and cropping methods, and (tilt) "
        f"aligns using OpticalFlow. Steps: {WORKFLOW}."
    )

    parser = argparse.ArgumentParser(
        parents=[base_parser], description=description
    )

    parser.add_argument(
        "inputfiles",
        nargs="+",
        help="List of sample stacks",
    )

    parser.add_argument(
        "--exclude_samples",
        nargs="+",
        help="List of samples images in the stack to be exlcuded. "
        "The number in the stack (not the angle) should be indicated ("
        "starting from 1).",
        required=False,
    )

    parser.add_argument(
        "--pixels_cut_from_borders",
        default=0,
        type=int,
        help="Number of pixels to cut the image from every border.",
        required=False,
    )

    parser.add_argument(
        "--align_2d_with_mask",
        action="store_true",
        help="If indicated, the 2D alignment between polarization (angle-angle)"
        " is going to be done using a mask (tanh of images) as intermediary.",
    )

    parser.add_argument(
        "--align_2d_with_roi",
        action="store_true",
        help="If indicated, the alignment will be done with a ROI and the given"
        " shift will be used for the entire images.",
    )

    parser.add_argument(
        "--tilt_align_with_mask",
        action="store_true",
        help="If indicated, the tilt alignment of the absorption and the signal"
        " stacks is going to be done using a mask (tanh of images) as "
        "intermediary",
    )

    parser.add_argument(
        "--align_2d_method",
        nargs=1,
        choices=[
            "crosscorr",
            "corrcoeff",
            "crosscorr-fourier",
            "of",
            "pyStackReg",
        ],
        type=str,
        default="crosscorr",
        help="Method used for the 2D alignment between polarizations.",
        required=False,
    )

    parser.add_argument(
        "--subpixel_2d_align",
        action="store_true",
        help=(
            "If indicated, the 2D alignment between polarizations will be "
            "at subpixel resolution."
        ),
    )

    parser.add_argument(
        "--roi_size_2d_align",
        type=float,
        default=0.8,
        help="ROI size from the center (in percentage, e.g. 0.5 or 0.8) for"
        "the alignment methods crosscorr and corrcoeff",
        required=False,
    )

    parser.add_argument(
        "--crop_method",
        choices=["cropping", "fill"],
        type=str,
        default="cropping",
        help="Choose if after the 2D alignment the image should be cropped "
        "or whether the pixels with value less than 0.1 should be filled "
        "with ones in order to create background and void problems with the "
        "natural logarithm.",
        required=False,
    )

    parser.add_argument(
        "--cropping_percentage",
        type=float,
        default=0.8,
        help="ROI size from the center (in percentage, e.g. 0.5 or 0.8) for"
        "the drift correction after alignment.",
        required=False,
    )

    parser.add_argument(
        "--tilt_align",
        action="store_true",
        help="If indicated, a final tilt align of absobrtion and signal stacks "
        "will be performed.",
    )

    parser.add_argument(
        "--tilt_align_alg",
        type=str,
        choices=["OF", "CTalign", "pyStackReg"],
        default="OF",
        help="Choose the algorithm used for final tilt alignment.",
        required=False,
    )

    parser.add_argument(
        "--nfid",
        type=int,
        default=30,
        help="Number of fiducials in images for tilt alignment with CTalign."
        "Default=30",
        required=False,
    )
    parser.add_argument(
        "--repeat_tilt_align",
        action="store_true",
        help="If indicated, repeat the final tilt align of absobrtion and "
        "signal stacks from file already processed.",
    )

    parser.add_argument(
        "--save_to_align",
        action="store_true",
        help="If indicated, save the images used in the 2D and tilt alignment, "
        "i.e., the masks and ROI selected.",
    )

    parser.add_argument(
        "--tilt_align_with_mag",
        action="store_true",
        help="If indicated, the tilt alignment of the absorption and the signal"
        " stacks is going to be done using a the magnetic signal as "
        "intermediary",
    )
    return parser


def process_polarizations(
    arguments, cut_pix, output_filename, exclude_samples=None
):
    """
    Method to isolate from the main programm the processing of the polarization
    stacks. The expected format is the one given by the magnetism pipeline.
    Given a cut_pix value different from zero, the image will be cutted the
    number of pixels from the border.
    """
    # Recover polarization stacks
    try:
        positive_polarization_file = next(
            file for file in arguments if "_1.0_" in file or "_1_" in file
        )
        negative_polarization_file = next(
            file for file in arguments if "_-1.0_" in file or "_-1_" in file
        )
    except Exception:
        raise Exception(
            "The input files must include (only) one positive "
            "polarisation stack and one negative polarisation stack."
        )

    # Recover angles and keep only valid ones, then construct angle-image pairs
    # An angle is valid if it exists in both polarisation stacks
    pos_pol_stack = h5py.File(positive_polarization_file, "r")
    neg_pol_stack = h5py.File(negative_polarization_file, "r")

    pos_pol_angles = list(
        map(round, pos_pol_stack["TomoNormalized"]["rotation_angle"])
    )
    neg_pol_angles = list(
        map(round, neg_pol_stack["TomoNormalized"]["rotation_angle"])
    )
    valid_angles = set(pos_pol_angles) & set(neg_pol_angles)

    pos_pol_angles_images = [
        (
            angle,
            image
            if cut_pix == 0
            else image[cut_pix:-cut_pix, cut_pix:-cut_pix],
        )
        for angle, image in zip(
            pos_pol_angles, pos_pol_stack["TomoNormalized"]["TomoNormalized"]
        )
        if angle in valid_angles
    ]
    neg_pol_angles_images = [
        (
            angle,
            image
            if cut_pix == 0
            else image[cut_pix:-cut_pix, cut_pix:-cut_pix],
        )
        for angle, image in zip(
            neg_pol_angles, neg_pol_stack["TomoNormalized"]["TomoNormalized"]
        )
        if angle in valid_angles
    ]

    pos_pol_angles_images = sorted(
        pos_pol_angles_images, key=lambda tup: tup[0]
    )
    neg_pol_angles_images = sorted(
        neg_pol_angles_images, key=lambda tup: tup[0]
    )

    pos_pol_stack.close()
    neg_pol_stack.close()

    # If indicated, some sample angles are excluded
    if exclude_samples is not None:
        print(f"Excluding samples with index {exclude_samples}.\n")
        exclude_samples = np.sort(exclude_samples)[::-1]
        for idx in exclude_samples:
            del pos_pol_angles_images[int(idx) - 1]
            del neg_pol_angles_images[int(idx) - 1]

    # Save original data
    with h5py.File(output_filename, "a") as results:
        results.create_dataset(
            "OriginalPositiveStack",
            data=np.array([img for _, img in pos_pol_angles_images]),
        )
        results.create_dataset(
            "OriginalNegativeStack",
            data=np.array([img for _, img in neg_pol_angles_images]),
        )
        results.create_dataset(
            "Angles",
            data=np.array([angle for angle, _ in pos_pol_angles_images]),
        )

    return pos_pol_angles_images, neg_pol_angles_images


def create_masks(images_stack, message=""):
    """
    Given an stack of images, it creates an histogram of the central image
    (the one closer to the 0 degs) and allows the user to select a threshold
    used to compute a binarization (used later for alignemnt). It also allows
    the selection of a sigma and whether to use the negative of the mask.
    """
    fig = plt.figure(
        figsize=(0, 0)
    )  # Without this the QInputDialog below gives error
    plt.close(fig)
    thresh_change_with_angle = (
        QMessageBox.question(
            None,
            "Threshold changes with angle",
            "Do you want to use the threshold of the mask to change with the "
            "angle?\n Select 'YES' for samples where the absorption changes "
            "with the angle due to the effective thickness",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        == QMessageBox.Yes
    )
    threshold = 0.7
    sigma = 25

    accepted_threshold = False
    accepted_sigma = False
    accepted_threshold = False
    init = True
    while not accepted_threshold or not accepted_sigma:
        if thresh_change_with_angle:
            masks = [
                (
                    angle,
                    (
                        -np.tanh(
                            sigma
                            * (img - threshold * np.cos(np.deg2rad(angle)))
                        )
                        + 1
                    )
                    / 2,
                )
                for angle, img in images_stack
            ]
        else:
            masks = [
                (angle, (-np.tanh(sigma * (img - threshold)) + 1) / 2)
                for angle, img in images_stack
            ]
        fig, ax = plt.subplots()
        ax.set_title("Visualization of the mask")

        ax.imshow(
            images_stack[int(len(images_stack) / 2)][1],
            cmap="viridis",
            alpha=0.8,
        )
        ax.imshow(masks[int(len(masks) / 2)][1], cmap="gray", alpha=0.5)
        plt.show(block=False)

        # Choose new threshold
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.hist(images_stack[int(len(images_stack) / 2)][1].ravel(), bins=75)
        if init:
            fig.suptitle(
                message + f" (Current default value is {'%.3f' % threshold}. "
                "Choose a new one\n or directly click enter to "
                "close the window if the current value is correct.)",
                fontsize=10,
            )
        else:
            fig.suptitle(
                message + f" (Current value is {'%.3f' % threshold}. "
                "Choose a new one\n or directly click enter to close "
                "the window if the current value is correct.)",
                fontsize=10,
            )
        init = False
        fig.show()
        new_threshold = fig.ginput(1, timeout=0)
        plt.close()

        if len(new_threshold) == 0:
            accepted_threshold = True
        else:
            threshold = new_threshold[0][0]

        # Choose new sigma
        new_sigma, _ = QInputDialog.getText(
            None,
            "Sigma for the mask",
            f"Current sigma is {sigma}. If you want to try a new "
            "one, introduce it; if not, leave it blank: ",
        )
        if new_sigma == "":
            accepted_sigma = True
        else:
            sigma = int(new_sigma)

        plt.close("all")
    plt.close("all")

    # Add Gaussian noise
    gaussian = np.abs(
        np.random.normal(
            loc=0, scale=0.01, size=(masks[0][1].shape[0], masks[0][1].shape[1])
        )
    )

    # Choose whether to use the negative of the masks.
    fig, ax = plt.subplots()
    ax.set_title("Visualization of the mask")
    ax.imshow(masks[int(len(masks) / 2)][1], cmap="gray")
    plt.show(block=False)

    negative_dialog = (
        QMessageBox.question(
            None,
            "Negative of the mask",
            "Do you want to use the negative of the mask?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        == QMessageBox.Yes
    )

    if negative_dialog:
        masks = [(angle, 1 - (img + gaussian)) for angle, img in masks]
    else:
        masks = [(angle, img + gaussian) for angle, img in masks]
    plt.close("all")

    return masks


def mean_stack(stack):
    """
    Given a stack of images computes the mean of the entire stack (axis=0).
    """
    return np.mean(np.array([img for _, img in stack]), axis=0)


def compute_closest_in_list(references, to_compare):
    """
    Given a list of reference numbers, it computes the closes number for each
    element w.r.t. the elements of the to_compare list. It return the index
    of the closest value in the to_compare list.
    """
    idx_closest_to_compare_per_reference = [
        np.abs(np.array([angle for angle, _ in to_compare]) - angle).argmin()
        for angle, _ in references
    ]

    return idx_closest_to_compare_per_reference


def decide_radial_mask_radius(ref_img):
    """
    Interactive radius selection using Matplotlib slider.

    Returns selected radius between 0 and 1.0.
    """
    # Create figure with space for slider
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    # Initial radius
    initial_radius = 0.8

    # Create filter and process image
    def update_filter(radius):
        if radius == 0:
            filter = np.ones(ref_img.shape)
        else:
            filter = maskRaisedCosineRadial(
                shape=ref_img.shape,
                radius=np.min(ref_img.shape) * radius / 2,
                pad=20,
            )

        filtered_ref_img = (ref_img - np.mean(ref_img)) * filter
        filtered_ref_img_fourier = np.fft.fftshift(
            np.fft.fft2(filtered_ref_img)
        )

        ax1.clear()
        ax2.clear()

        ax1.set_title("Real Space")
        ax1.imshow(filtered_ref_img, cmap="gray", aspect="auto")

        ax2.set_title("Fourier Space")
        ax2.imshow(
            np.log(abs(filtered_ref_img_fourier)), cmap="gray", aspect="auto"
        )

        fig.suptitle(
            "Exemplary effect of the radial filter over the central "
            f"image with r = {radius*100:.1f}% shape"
        )
        # f"Radial Filter Effect (r = {radius*100:.1f}% shape)")
        fig.canvas.draw_idle()

    # Slider for radius
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    radius_slider = Slider(ax_slider, "Radius", 0, 1.0, valinit=initial_radius)

    # Update plot when slider changes
    radius_slider.on_changed(update_filter)

    # Initial plot
    update_filter(initial_radius)

    # Store selected radius
    selected_radius = [initial_radius]

    # Close and store radius on key press
    def on_key(event):
        if event.key == "enter":
            selected_radius[0] = radius_slider.val
            plt.close(fig)

    # Store radius if window is closed
    def on_close(event):
        selected_radius[0] = radius_slider.val

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    plt.show(block=True)

    return selected_radius[0]


def align_2d_polarizations(pol_pairs, align_info):
    """
    Align the negative polarization image using the positive one as a reference.
    It returns the aligned positive and negative polarization images,
    its angle, and the used movement vector.
    Two alignment methods are available:
        - Cross-correlation using cv2.TM_CCOEFF_NORMED,
        - Optical Flow (xpy_tilt_align with ref=0)
    :return: Angle, aligned polarization images, and the applied movement vector
    """
    angle, pos_pol_image = pol_pairs[0]
    _, neg_pol_image = pol_pairs[1]

    # Determine movement method
    if align_info["subpixel"]:
        mv_projection_m = mv_projection_subpixel
    else:
        mv_projection_m = mv_projection

    if align_info["method"] in ("crosscorr", "corrcoeff"):
        # Cross-correlation or Correlation coefficient methods
        aligned_pos_pol_image = pos_pol_image
        aligned_neg_pol_image, mv_vector = align(
            image_ref=pos_pol_image,
            image_to_align=neg_pol_image,
            align_method=(
                "cv2.TM_CCOEFF_NORMED"
                if align_info["method"] == "corrcoeff"
                else "cv2.TM_CCORR_NORMED"
            ),
            roi_size=align_info["roi_size"],
        )
        aligned_neg_pol_image = mv_projection_m(
            image=neg_pol_image, mv_vector=mv_vector, fill_val=1
        )
        mv_vector = [(0, 0), mv_vector]
    elif align_info["method"] == "crosscorr-fourier":
        # Since it works in Fourier we need to smooth the borders of the image
        # by using a pixel-wise filter. Additionally, we delete zero-order
        # peaks: img = (img - mean(img)) * maks
        if align_info["radius"] == 0:
            filter = np.ones(pos_pol_image.shape)
        else:
            filter = maskRaisedCosineRadial(
                shape=pos_pol_image.shape,
                radius=np.min(pos_pol_image.shape) * align_info["radius"] / 2,
                pad=20,
            )
        pos_pol_image_filtered = (
            pos_pol_image - np.mean(pos_pol_image)
        ) * filter
        neg_pol_image_filtered = (
            neg_pol_image - np.mean(neg_pol_image)
        ) * filter
        mv_vector, _, _ = phase_cross_correlation(
            reference_image=pos_pol_image_filtered,
            moving_image=neg_pol_image_filtered,
            space="real",
            upsample_factor=500 if align_info["subpixel"] else 1,
        )
        mv_vector = list(map(np.float32, mv_vector))

        aligned_pos_pol_image = pos_pol_image
        try:
            aligned_neg_pol_image = mv_projection_m(
                image=neg_pol_image, mv_vector=mv_vector, fill_val=1
            )
        except Exception:
            print(f"Does not worked: {mv_vector}")
            aligned_neg_pol_image = neg_pol_image
        mv_vector = [(0, 0), mv_vector]
    elif align_info["method"] == "of":
        # Optical flow method
        # Apply filter for a better usage of OF in Fourier space
        if align_info["radius"] == 0:
            filter = np.ones(pos_pol_image.shape)
        else:
            filter = maskRaisedCosineRadial(
                shape=pos_pol_image.shape,
                radius=np.min(pos_pol_image.shape) * align_info["radius"] / 2,
                pad=20,
            )
        pos_pol_image_filtered = (
            pos_pol_image - np.mean(pos_pol_image)
        ) * filter
        neg_pol_image_filtered = (
            neg_pol_image - np.mean(neg_pol_image)
        ) * filter

        tmp_stack_to_align = np.array(
            [pos_pol_image_filtered, neg_pol_image_filtered]
        )

        _, mv_vector = tilt_align(
            stk=tmp_stack_to_align,
            tiltList=None,
            refId=0,
            centerStr=f"{align_info['center'][0]},{align_info['center'][1]}",
            xRange=align_info["1"][1] - align_info["0"][1],
            yRange=align_info["1"][0] - align_info["0"][0],
        )
        mv_vector = [
            ((mv_vector[0][1][-1]), (mv_vector[0][0][-1])),
            ((mv_vector[1][1][-1]), (mv_vector[1][0][-1])),
        ]
        # Align images using the movement vectors
        aligned_pos_pol_image = mv_projection_m(
            image=pos_pol_image, mv_vector=mv_vector[0], fill_val=1
        )

        aligned_neg_pol_image = mv_projection_m(
            image=neg_pol_image, mv_vector=mv_vector[1], fill_val=1
        )
    elif align_info["method"] == "pyStackReg":
        sr = StackReg(StackReg.TRANSLATION)
        aligned_pos_pol_image = pos_pol_image
        tform = sr.register(aligned_pos_pol_image, neg_pol_image)
        mv_vector = [(0, 0), (-(tform[1, 2]), -(tform[0, 2]))]
        aligned_neg_pol_image = mv_projection_m(
            image=neg_pol_image, mv_vector=mv_vector[1], fill_val=1
        )
    else:
        raise Exception(
            f"2D alignment method '{align_info['method']}' is not valid."
        )

    return angle, aligned_pos_pol_image, aligned_neg_pol_image, mv_vector


def align_2d(
    args,
    pos_pol_angles_images,
    neg_pol_angles_images,
    output_filename,
):
    """
    Manages the aligment between polarization stacks allowing to use a mask
    and/or a ROI. Returns and saves the result of the alignment.
    """
    method2d = args.align_2d_method[0]
    x0 = y0 = x1 = y1 = center_x = center_y = None

    print(
        "Aligning 2D polarization images"
        + f"{' (subpixel)' if args.subpixel_2d_align else ''} using"
        + (" a mask" if args.align_2d_with_mask else "")
        + (" a selected ROI" if args.align_2d_with_roi else "")
        + (" and" if args.align_2d_with_mask or args.align_2d_with_roi else "")
        + f" method {args.align_2d_method[0]}.\n"
    )

    pos_pol_to_align = pos_pol_angles_images
    neg_pol_to_align = neg_pol_angles_images
    if args.align_2d_with_mask:
        # Create masks for both polarizations
        pos_pol_to_align = create_masks(
            pos_pol_to_align,
            message="Select threshold for positive polarization mask",
        )
        neg_pol_to_align = create_masks(
            neg_pol_to_align,
            message="Select threshold for negative polarization mask",
        )
        plt.close()

    if args.align_2d_with_roi:
        # Select ROI and crop images to align the ROIs and then use the shifts
        # for the entire images
        mean_to_align = mean_stack(pos_pol_to_align + neg_pol_to_align)

        roi_to_align = select_roi(mean_to_align, step="Select ROI to align")
        x0, y0 = roi_to_align[-2]
        x1, y1 = roi_to_align[-1]

        pos_pol_to_align = [
            (angle, image[x0:x1, y0:y1]) for angle, image in pos_pol_to_align
        ]
        neg_pol_to_align = [
            (angle, image[x0:x1, y0:y1]) for angle, image in neg_pol_to_align
        ]

    # Save data used in the alignment
    if args.save_to_align:
        file_to_align = os.path.join(
            os.path.dirname(output_filename), "toAlign.hdf5"
        )
        with h5py.File(file_to_align, "w") as f:
            f.create_dataset(
                "ToAlign2DNegativePol",
                data=np.array([img for _, img in neg_pol_to_align]),
            )
            f.create_dataset(
                "ToAlign2DPositivePol",
                data=np.array([img for _, img in pos_pol_to_align]),
            )
    if method2d == "of":
        # Request ROI to focus the alignment usign OF
        mean_pol_masks = mean_stack(pos_pol_to_align + neg_pol_to_align)

        align_2d_roi = select_roi(
            np.array(mean_pol_masks),
            step="Samples center for the 2D alignment",
        )
        x0, y0 = align_2d_roi[-2]
        x1, y1 = align_2d_roi[-1]
        center_x, center_y = int((x0 + x1) / 2), int((y0 + y1) / 2)

    radius = None
    if method2d in ("crosscorr-fourier", "of"):
        # Look for the best radius for the filter
        ref_img = pos_pol_to_align[int(len(pos_pol_to_align) / 2)][1]
        radius = decide_radial_mask_radius(ref_img)

    alignment_info = {
        "method": method2d,
        "0": (x0, y0),
        "1": (x1, y1),
        "center": (center_x, center_y),
        "roi_size": args.roi_size_2d_align,
        "radius": radius,
        "subpixel": args.subpixel_2d_align,
    }
    arguments = [
        [(pos_pol, neg_pol), alignment_info]
        for pos_pol, neg_pol in zip(pos_pol_to_align, neg_pol_to_align)
    ]
    with Parallel(n_jobs=1) as parallel:
        outputs = parallel(
            delayed(align_2d_polarizations)(*args) for args in arguments
        )

    # Sort outputs
    outputs = sorted(outputs, key=lambda e: e[0])

    # Recover (aligned) images
    if args.align_2d_with_mask or args.align_2d_with_roi:
        # The output cannot be used for both cases and the shift is recovered
        # to align the correct images
        outputs = [(angle, mv_vectors) for angle, _, _, mv_vectors in outputs]

        pos_images = [
            img for _, img in sorted(pos_pol_angles_images, key=lambda e: e[0])
        ]
        neg_images = [
            img for _, img in sorted(neg_pol_angles_images, key=lambda e: e[0])
        ]
        pos_neg_images = list(zip(pos_images, neg_images))

        to_align = [
            (angle, mv_vectors, pos_img, neg_img)
            for (angle, mv_vectors), (pos_img, neg_img) in zip(
                outputs, pos_neg_images
            )
        ]
        pos_pol_angles_images = [
            (
                angle,
                mv_projection(
                    image=np.array(pos_img), mv_vector=mv[0], fill_val=1
                ),
            )
            for angle, mv, pos_img, _ in to_align
        ]
        neg_pol_angles_images = [
            (
                angle,
                mv_projection(
                    image=np.array(neg_img), mv_vector=mv[1], fill_val=1
                ),
            )
            for angle, mv, _, neg_img in to_align
        ]
    else:
        pos_pol_angles_images = [
            (angle, pos_img) for angle, pos_img, _, _ in outputs
        ]
        neg_pol_angles_images = [
            (angle, neg_img) for angle, _, neg_img, _ in outputs
        ]

    # Save alignment data
    with h5py.File(output_filename, "a") as results:
        results.create_dataset(
            "2DAlignedPositiveStack",
            data=np.array([img for _, img in pos_pol_angles_images]),
        )
        results.create_dataset(
            "2DAlignedNegativeStack",
            data=np.array([img for _, img in neg_pol_angles_images]),
        )

    return pos_pol_angles_images, neg_pol_angles_images


def solve_drift_alignment(args, pos_pol_angles_images, neg_pol_angles_images):
    """
    After the alignment of two images, a drift is created in the results.
    This can be solve by cropping or filling with ones the areas without
    information using this method.
    """

    if args.crop_method == "cropping":
        # Show average of all images in polarizations (for a better
        # understanding of the sample distribution over the field of view)
        # If the cropping is not consistent for all images divisions by zero
        # could appear latter while applying the natural logarithm
        print("\nCropping images after alignment.\n")

        h, w = np.array(pos_pol_angles_images[0][1]).shape[:2]
        crop_h = int(h * args.cropping_percentage)
        crop_w = int(w * args.cropping_percentage)

        x0 = (h - crop_h) // 2
        y0 = (w - crop_w) // 2

        # Average images for a better exemplary visualization throught the stack
        # of the zone that it's going to be cropped
        mean_img = mean_stack(pos_pol_angles_images + neg_pol_angles_images)
        fig, ax = plt.subplots()
        ax.set_title(
            "Exemplary visualization of the (mean) stack after"
            "cropping to correct post-alignment drift"
        )
        ax.imshow(mean_img[x0 : x0 + crop_h, y0 : y0 + crop_w], cmap="gray")
        plt.show(block=False)

        # Cropping images
        pos_pol_angles_images = [
            (angle, image[x0 : x0 + crop_h, y0 : y0 + crop_w])
            for angle, image in pos_pol_angles_images
        ]
        neg_pol_angles_images = [
            (angle, image[x0 : x0 + crop_h, y0 : y0 + crop_w])
            for angle, image in neg_pol_angles_images
        ]
    elif args.crop_method == "fill":
        # Fill all pixels with value < threshold with mean of a ROI from central
        # image to recreate the background
        threshold = 0.005
        print(
            f"\nFilling pixels with values below {threshold} with mean values "
            "of ROIs from the central image after cropping.\n"
        )

        crop_roi = select_roi(
            pos_pol_angles_images[int(len(pos_pol_angles_images) / 2)][1],
            step="Select sample image ROI of the central image for the positive"
            " stack for the computation of the mean value to fill pixels.",
        )
        x0_pos, y0_pos = crop_roi[-2]
        x1_pos, y1_pos = crop_roi[-1]

        crop_roi = select_roi(
            neg_pol_angles_images[int(len(neg_pol_angles_images) / 2)][1],
            step="Select sample image ROI of the central image for the negative"
            " stack for the computation of the mean value to fill pixels.",
        )
        x0_neg, y0_neg = crop_roi[-2]
        x1_neg, y1_neg = crop_roi[-1]

        mean_value_pos = np.mean(
            pos_pol_angles_images[int(len(pos_pol_angles_images) / 2)][1][
                x0_pos:x1_pos, y0_pos:y1_pos
            ]
        )
        mean_value_neg = np.mean(
            neg_pol_angles_images[int(len(neg_pol_angles_images) / 2)][1][
                x0_neg:x1_neg, y0_neg:y1_neg
            ]
        )

        accepted_threshold = False
        while not accepted_threshold:
            fig, ax = plt.subplots()
            ax.set_title("Fill threshold preview (only positive stack)")
            ax.imshow(
                np.where(
                    pos_pol_angles_images[0][1] < threshold,
                    mean_value_pos,
                    pos_pol_angles_images[0][1],
                ),
                cmap="gray",
            )
            plt.show(block=False)

            new_thres, _ = QInputDialog.getText(
                None,
                "Threshold for filling",
                "Current threshold for the filling with ones "
                f"after aligning is {threshold}. "
                "If you want to try a new one, introduce it; "
                "otherwise, leave it blank: ",
            )

            if new_thres == "":
                accepted_threshold = True
            else:
                threshold = float(new_thres)
            plt.close()
        plt.close("all")

        pos_pol_angles_images = [
            (angle, np.where(image < threshold, mean_value_pos, image))
            for angle, image in pos_pol_angles_images
        ]
        neg_pol_angles_images = [
            (angle, np.where(image < threshold, mean_value_neg, image))
            for angle, image in neg_pol_angles_images
        ]
    else:
        raise Exception(
            f"The chosen cropping method '{args.crop_method}' is not valid."
        )

    return pos_pol_angles_images, neg_pol_angles_images


def intensity_correction(pos_pol_angles_images, neg_pol_angles_images):
    """
    It computes an intensity correction factor based on a ROI from both
    polarizations and then applies the factor of each image to the
    negative stack.
    """

    msg_box = QMessageBox()
    msg_box.setWindowTitle(
        "Do you want to apply an intensity correction factor?"
    )
    msg_box.setText(
        "Do you want to apply an intensity correction factor "
        "to the negative images?"
    )

    ok_button = msg_box.addButton("Yes", QMessageBox.AcceptRole)
    _ = msg_box.addButton("No", QMessageBox.RejectRole)

    msg_box.exec_()

    if msg_box.clickedButton() == ok_button:
        print(
            "\nApplying intensity correction factor to the "
            "negative polarization.\n"
        )

        mean_imgs = mean_stack(pos_pol_angles_images + neg_pol_angles_images)

        crop_roi = select_roi(
            mean_imgs,
            step="Select sample image ROI for the computation "
            "of the intensity correction factor",
        )

        x0, y0 = crop_roi[-2]
        x1, y1 = crop_roi[-1]

        factors = [
            np.mean(pos_angle_image[1][x0:x1, y0:y1])
            / np.mean(neg_angle_image[1][x0:x1, y0:y1])
            for pos_angle_image, neg_angle_image in zip(
                pos_pol_angles_images, neg_pol_angles_images
            )
        ]

        neg_pol_angles_images = [
            (neg_angle_image[0], neg_angle_image[1] * factor)
            for factor, neg_angle_image in zip(factors, neg_pol_angles_images)
        ]

    return pos_pol_angles_images, neg_pol_angles_images


def ln(angle_img):
    """
    Given an image, it returns the negative natural logarithm of it,
    except if the image is problematic, i.e., it has zeros/NaNs.
    """
    problematic_image_found = False

    try:
        return (angle_img[0], -np.log(angle_img[1], where=(angle_img[1] != 0)))
    except Exception:
        problematic_image_found = True
        print(
            f"Image {angle_img[0]} has zeros and the logarithm"
            "has not been computed.\n"
        )

    if problematic_image_found:
        raise Exception(
            "Some images remains with zeros and it cannot continue.\n"
        )


def stack_tilt_align(
    absorption,
    signal,
    tilt_align_with_mask,
    output_filename,
    algorithm="OF",
    nom_angles=None,
    nfid=30,
    save_to_align=False,
    tilt_align_with_mag=False,
):
    """
    Perform tilt alignment of the absorption stack and apply the transformations
    to the signal stack. It can be done using a mask if tilt_align_with_mask
    is set to true.
    There are two algorithms for tilt alignment: OF (Optical Flow)
    or CTalign (IMOD). For CTalign it is necessary to introduce the nom_angles.
    The argument nfid indicates the number of fiducials in the images.
    """
    if tilt_align_with_mag:
        tmp_stack_to_align = signal
    else:
        tmp_stack_to_align = absorption
    if algorithm == "OF":
        # Apply filter for a better usage of OF in Fourier space
        ref_img = tmp_stack_to_align[int(len(tmp_stack_to_align) / 2)][1]
        radius = decide_radial_mask_radius(ref_img)

        filter = maskRaisedCosineRadial(
            shape=tmp_stack_to_align[0][1].shape,
            radius=np.min(tmp_stack_to_align[0][1].shape) * radius / 2,
            pad=20,
        )
        tmp_stack_to_align = [
            (angle, (image - np.mean(image)) * filter)
            for angle, image in tmp_stack_to_align
        ]

        # Use mask for the alignment if requested

        if tilt_align_with_mask:
            print("A mask is going to be used for the tilt alignment.\n")
            masks = create_masks(tmp_stack_to_align)

        # Save the filtered set in a HDF5
        to_ali = masks if tilt_align_with_mask else tmp_stack_to_align
        tmp_stack_to_align = np.array([image for _, image in to_ali])

        # Select ROI to compute the center of the alignment and the range
        # covered in x and y
        mean_stack = np.mean(
            np.array([img for img in tmp_stack_to_align]), axis=0
        )
        stack_roi = select_roi(
            np.array(mean_stack), step="Sample center for the 3D alignment"
        )
        x0, y0 = stack_roi[-2]
        x1, y1 = stack_roi[-1]
        center_x, center_y = int((x0 + x1) / 2), int((y0 + y1) / 2)

        # Select radius for the tilt align
        radius = 32
        new_radius, _ = QInputDialog.getText(
            None,
            "Radius for tilt alignment",
            f"\nCurrent radius for the OpticalFLow is {radius}."
            " If you want to try a new one, introduce it;"
            " if not, leave it blank: ",
        )
        if new_radius != 32 and new_radius != "":
            radius = int(new_radius)

        print("\n")

        _, mv_vectors = tilt_align(
            stk=tmp_stack_to_align,
            tiltList=None,
            refId=0,
            centerStr=f"{center_x},{center_y}",
            xRange=x1 - x0,
            yRange=y1 - y0,
            radius=radius,
            # nProcs=1
        )

        # Align absorption (used mv_projection in case mask was used) and signal
        aligned_absorption = [
            mv_projection(
                image=absorption_img,
                mv_vector=(int(mv_vector[1][-1]), int(mv_vector[0][-1])),
            )
            for absorption_img, mv_vector in zip(
                [absorption_img for _, absorption_img in absorption],
                mv_vectors,
            )
        ]

        aligned_signal = [
            mv_projection(
                image=signal_image,
                mv_vector=(int(mv_vector[1][-1]), int(mv_vector[0][-1])),
            )
            for signal_image, mv_vector in zip(
                [signal_image for _, signal_image in signal], mv_vectors
            )
        ]

    elif algorithm == "CTalign":
        path = "ctalignxcorr"
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        hdf5_file = os.path.join("toAlign.hdf5")
        mrc_file_ali = os.path.join("toAlign.mrc")
        mrc_file_absorption = os.path.join("absorption.mrc")
        mrc_file_XMCD = os.path.join("magneticSignal.mrc")

        # Create files for alignment
        np.savetxt(os.path.join("toAlign.tlt"), nom_angles, fmt="%.4f")
        if tilt_align_with_mask:
            print("A mask is going to be used for the tilt alignment.\n")
            tmp_stack_to_align = create_masks(tmp_stack_to_align)
        tmp_stack_to_align = np.array(
            [
                image
                + np.abs(
                    np.random.normal(
                        np.mean(image), np.std(image), np.shape(image)
                    )
                )
                for _, image in tmp_stack_to_align
            ]
        )
        absorption = np.array([image for _, image in absorption])
        signal = np.array([image for _, image in signal])
        with h5py.File(hdf5_file, "w") as f:
            f.create_dataset(
                "TomoNormalized/TomoNormalized", data=tmp_stack_to_align
            )
            f.create_dataset("TomoNormalized/rotation_angle", data=nom_angles)
        with mrcfile.new(mrc_file_ali, overwrite=True) as f:
            f.set_data(tmp_stack_to_align.astype(np.float32))
        with mrcfile.new(mrc_file_absorption, overwrite=True) as f:
            f.set_data(absorption.astype(np.float32))
        with mrcfile.new(mrc_file_XMCD, overwrite=True) as f:
            f.set_data(signal.astype(np.float32))

        # Solve alignment
        cmd = (
            f"ctalignxcorr {hdf5_file} {mrc_file_ali} --tilt_option 2 "
            f"--n_fid {nfid} > ctalignxcorr.log"
        )
        os.system(cmd)

        # Read .tlt and write them in hdf5
        tlt_file = mrc_file_ali.replace("mrc", "tlt")
        angles = np.loadtxt(tlt_file)
        with h5py.File(output_filename, "a") as f:
            f.create_dataset("NominalAngles", data=nom_angles)
            del f["Angles"]
            f.create_dataset("Angles", data=angles)

        # Read transformations (*.xf) and apply them to XMCD
        xf_file = mrc_file_ali.replace("mrc", "xf")
        cmd = (
            f"newstack -input {mrc_file_absorption} "
            f"-output ali{mrc_file_absorption} -offset 0,0 -xform {xf_file} "
            f"-origin -taper 1,0 >> ctalignxcorr.log"
        )
        os.system(cmd)
        cmd = (
            f"newstack -input {mrc_file_XMCD} -output ali{mrc_file_XMCD} "
            f"-offset 0,0 -xform {xf_file} -origin "
            f"-taper 1,0 >> ctalignxcorr.log"
        )
        os.system(cmd)

        # Read MRC files
        with mrcfile.open(f"ali{mrc_file_absorption}") as f:
            aligned_absorption = f.data
        with mrcfile.open(f"ali{mrc_file_XMCD}") as f:
            aligned_signal = f.data
        os.chdir("../")

    elif algorithm == "pyStackReg":
        if tilt_align_with_mask:
            print("A mask is going to be used for the tilt alignment.\n")
            tmp_stack_to_align = create_masks(tmp_stack_to_align)
        tmp_stack_to_align = np.array(
            [image for _, image in tmp_stack_to_align]
        )
        absorption = np.array([image for _, image in absorption])
        signal = np.array([image for _, image in signal])
        from pystackreg import StackReg

        sr = StackReg(StackReg.TRANSLATION)
        _ = sr.register_stack(tmp_stack_to_align, reference="previous")
        aligned_absorption = sr.transform_stack(absorption)
        aligned_signal = sr.transform_stack(signal)

    # Save results
    if save_to_align:
        file_to_align = os.path.join(
            os.path.dirname(output_filename), "toAlign.hdf5"
        )
        with h5py.File(file_to_align, "a") as f:
            if "ToTiltAlign" in f:
                del f["ToTiltAlign"]
            f.create_dataset("ToTiltAlign", data=np.array(tmp_stack_to_align))

    with h5py.File(output_filename, "a") as results:
        results.create_dataset(
            "AbsorptionTiltAligned", data=np.array(aligned_absorption)
        )
        results.create_dataset(
            "MagneticSignalTiltAligned", data=np.array(aligned_signal)
        )


def main(args=None):
    parser = app_parser()
    args = parser.parse_args(args)

    start_time = time.time()

    # Decide output filename
    dir_file, file = os.path.split(args.inputfiles[0])
    output_filename = os.path.join(dir_file, f"{file.split('_')[0]}_xmcd.hdf5")
    if not args.repeat_tilt_align:
        # Check if a previous results file exists. If so, delelte it
        if os.path.isfile(output_filename):
            print(
                f"File {output_filename} exists from a previous execution and "
                "it is going to be deleted.\n"
            )
            os.remove(output_filename)

        # Recover sample stacks
        pos_pol_angles_images, neg_pol_angles_images = process_polarizations(
            args.inputfiles,
            int(args.pixels_cut_from_borders),
            output_filename,
            args.exclude_samples,
        )

        # Align images from both polarizations 1:1 by angles in 2D
        # Alignment can be done using a mask (image binarization) or not
        pos_pol_angles_images, neg_pol_angles_images = align_2d(
            args, pos_pol_angles_images, neg_pol_angles_images, output_filename
        )

        # Solve the drift caused while aligning in 2D
        pos_pol_angles_images, neg_pol_angles_images = solve_drift_alignment(
            args, pos_pol_angles_images, neg_pol_angles_images
        )

        # Apply intensity compensation factors between polarizations after
        # selecting ROI over the average of sample images (for a better
        # understanding of the sample distribution over the field of view)
        pos_pol_angles_images, neg_pol_angles_images = intensity_correction(
            pos_pol_angles_images, neg_pol_angles_images
        )

        # Apply the negative of the natural logarithm
        print(
            "Applying the negative of the natural log to both polarizations.\n"
        )
        log_pos_pol_angles_images = list(map(ln, pos_pol_angles_images))
        log_neg_pol_angles_images = list(map(ln, neg_pol_angles_images))

        # Replace nan values with 0
        log_pos_pol_angles_images = [
            (ang, np.where(np.isnan(img), 0, img))
            for ang, img in log_pos_pol_angles_images
        ]
        log_neg_pol_angles_images = [
            (ang, np.where(np.isnan(img), 0, img))
            for ang, img in log_neg_pol_angles_images
        ]

        # Compute absorbence (+) and magnetic or dichroic signal/XMCD (-)
        print("Computing absorption and signal stacks.\n")
        absorption = [
            (angle_img_pos[0], (angle_img_pos[1] + angle_img_neg[1]) / 2)
            for angle_img_pos, angle_img_neg in zip(
                log_pos_pol_angles_images, log_neg_pol_angles_images
            )
        ]
        signal = [
            (angle_img_pos[0], (angle_img_pos[1] - angle_img_neg[1]) / 2)
            for angle_img_pos, angle_img_neg in zip(
                log_pos_pol_angles_images, log_neg_pol_angles_images
            )
        ]

        # Save absorption and magnetic signal stacks
        with h5py.File(output_filename, "a") as results:
            results.create_dataset(
                "Absorption2DAligned",
                data=np.array([charge_img for _, charge_img in absorption]),
            )
            results.create_dataset(
                "MagneticSignal2DAligned",
                data=np.array([signal_img for _, signal_img in signal]),
            )

    if args.tilt_align:
        # When repeating tilt align, check if necessary variables exists,
        # if not, read the datsets.
        if "absorption" not in locals():
            with h5py.File(output_filename, "a") as f:
                absorption = f["Absorption2DAligned"][...]
                signal = f["MagneticSignal2DAligned"][...]
                angles = f["Angles"][...]
                # Erase tilt aligned datasets
                if "AbsorptionTiltAligned" in f:
                    del f["AbsorptionTiltAligned"]
                if "MagneticSignalTiltAligned" in f:
                    del f["MagneticSignalTiltAligned"]
            absorption = [
                (angle, img) for angle, img in zip(angles, absorption)
            ]
            signal = [(angle, img) for angle, img in zip(angles, signal)]
        # Tilt alignment of the absorption stack and the signal stack
        # Since the signal cannot be aligned, the absorption is firstly aligned
        # and the given movement vectors are used to align the signal (this
        # is possible thanks to the 2D alignment between polarizations that has
        # been previously performed)
        print("Tilt aligning absorption and signal.\n")

        stack_tilt_align(
            absorption=absorption,
            signal=signal,
            tilt_align_with_mask=args.tilt_align_with_mask,
            output_filename=output_filename,
            algorithm=args.tilt_align_alg,
            nom_angles=[a[0] for a in signal],
            nfid=args.nfid,
            save_to_align=args.save_to_align,
            tilt_align_with_mag=args.tilt_align_with_mag,
        )

    # Close all matplotlib windows to avoid raising false errors
    plt.close("all")

    print(f"magnetism_xmcd took {time.time() - start_time} seconds\n")


if __name__ == "__main__":
    main()
