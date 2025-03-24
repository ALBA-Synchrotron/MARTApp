#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2023 ALBA Synchrotron
#
# Authors: A. Estela Herguedas Alonso, Joaquin Gomez Sanchez
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
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pystackreg import StackReg
from scipy.optimize import least_squares
from skimage import transform

from sdm.mistral.scripts.parser import base_parser
from sdm.mistral.util import select_roi

WORKFLOW = "cosine_stretching >> join_tilt_series >> reconstruction"


def app_parser():
    """
    Defnes parser arguments for the magnetism_2Dreconstruction pipeline.
    """
    description = (
        "This pipeline computes the reconstruction of the 3D magnetization of "
        f"2D magnetic structures. Steps: {WORKFLOW}"
    )

    parser = argparse.ArgumentParser(
        parents=[base_parser], description=description
    )

    parser.add_argument(
        "stack",
        type=str,
        help="Output file (HDF5) from the magnetism_xmcd pipeline "
        "containing the stack.",
    )

    parser.add_argument(
        "rotated_stack",
        type=str,
        help="Output file (HDF5) from the magnetism_xmcd pipeline "
        "containign the rotated stack.",
    )

    parser.add_argument(
        "--crop_cos",
        action="store_true",
        help="If set, the user will be able to crop from stack the ROI for "
        "the registration done during the cosine stretching step.",
    )

    parser.add_argument(
        "--crop_rot_cos",
        action="store_true",
        help="If set, the user will be able to crop from the rotated stack "
        "the ROI for the registration done during the cosine stretching step.",
    )

    parser.add_argument(
        "--select_points_cos",
        action="store_true",
        help="If set, the initial transformation will be obtained from the "
        "selected points. Otherwise, it will be computed only from the angles.",
    )

    parser.add_argument(
        "--select_points_rot_cos",
        action="store_true",
        help="If set, the initial transformation for the rotated stack "
        "will be obtained from the selected points. "
        "Otherwise, it will be computed only from the angles.",
    )

    parser.add_argument(
        "--metric_cos",
        type=str,
        nargs=1,
        choices=["meansquares", "correlation"],
        default="meansquares",
        help="Metric to be reduced during the registration for the cosine "
        "stetching step of the stack.",
        required=False,
    )

    parser.add_argument(
        "--metric_rot_cos",
        type=str,
        nargs=1,
        choices=["meansquares", "correlation"],
        default="meansquares",
        help="Metric to be reduced during the registration for the cosine "
        "stetching step of the rotated stack.",
        required=False,
    )

    parser.add_argument(
        "--crop_join_tilt",
        action="store_true",
        help="If set, the user will be able to crop from the cosine stretching "
        "stacks in order to choose a ROI for the registration done during the "
        "join tilt series step.",
    )

    parser.add_argument(
        "--select_points_join_tilt",
        action="store_true",
        help="If set, the initial transformation for the join tilt series step "
        "will be obtained from the selected points. "
        "Otherwise, it will be computed only from the angles.",
    )

    parser.add_argument(
        "--initial_angle_join_tilt",
        type=float,
        default=90,
        help="Initial angle (in degrees) for the estimation of the "
        "transformation for the join tilt series step. Default value: 90",
    )

    parser.add_argument(
        "--metric_join_tilt",
        type=str,
        nargs=1,
        choices=["meansquares", "correlation"],
        default="meansquares",
        help="Metric to be reduced during the registration for the join tilt "
        "series step.",
        required=False,
    )

    parser.add_argument(
        "--attenuation_lenght",
        type=float,
        default=1,
        help="Value for attenuation length of magnetic material in meters. "
        "This value is applied for all the reconstructed volume. "
        "It can be obtained from the CXRO database. If unknown, set to 1.",
        required=False,
    )

    parser.add_argument(
        "--thickness",
        type=float,
        default=1,
        help="Value of the thickness of the magnetic material layer for "
        "reconstruction in meters. If unknown, set to 1.",
        required=False,
    )

    parser.add_argument(
        "--dichroic_coefficient",
        type=float,
        default=1,
        help="Dichroic coefficient. It can be obtained from the energy "
        "spectra, by computing the asymmetry ratio. If unknown, set to 1.",
        required=False,
    )

    parser.add_argument(
        "--repeat_step",
        type=int,
        default=0,
        help="Repeat a step: 1: cosine stretching tilt series 1, "
        "2: cosine stretching tilt series 2, 3: join tilt series, "
        "4: reconstruction. Default 0: repeat all",
        required=False,
    )

    parser.add_argument(
        "--distance_fiducials",
        type=float,
        default=0.0,
        help="Distance in pixels from the fiducials to the magnetic material.",
        required=False,
    )

    parser.add_argument(
        "--rot_axis",
        type=str,
        default="Y",
        choices=["X", "Y"],
        help="Rotation axis of the XMCD images.",
        required=False,
    )
    return parser


def point_selection(img1, img2, type="affine", msg=""):
    """
    Given two images, allows the user to select the same points in similar
    images and returns the affine transformation between them.
    The transformation is from img2 to img1 with the origin at
    the center of img1.

    Parameters:
    -----------
    img1 : ndarray, shape (x, y)
        2D array representing the fixed image.
    img2 : ndarray, shape (x, y)
        2D array representing the moving image.
    transformation_type : str, optional
        Type of geometric transformation to return.
        Options: 'affine' (default), 'rigid'.
    msg : str, optional
        Title to add in the point selection figure.

    Returns:
    --------
    M : list, shape (6,)
        1x6 list representing the affine transformation between img1 and img2.
    p1 : ndarray, shape (n, 2)
        2D array containing the points selected in img1.
    p2 : ndarray, shape (n, 2)
        2D array containing the corresponding points selected in img2.
    """
    ps = PointSelectionMain(img1, img2, msg=msg)
    ps.run()
    p1, p2 = ps.get_selected_points()
    M = ps.get_transform_matrix(type=type)
    return M, np.array(p1), np.array(p2)


class PointSelectionMain:
    def __init__(self, img1, img2, msg=""):
        self.img1 = img1
        self.img2 = img2
        self.points1 = []
        self.points2 = []

        self.fig, self.ax = plt.subplots(1, 2)
        self.text_ax = self.fig.add_axes([0.1, 0.9, 0.8, 0.05])
        self.text_ax.set_axis_off()
        self.text_ax.text(
            0.5,
            0.5,
            (
                "Points are selected with the right button on the mouse.\n"
                "Select the points in the same order in both figures.\n"
                "When finished, close the window.\n"
                f"{msg}"
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        self.cid1 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_img1
        )
        self.cid2 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_img2
        )

    def onclick_img1(self, event):
        if event.inaxes == self.ax[0] and event.button == 3:
            self.points1.append((event.xdata, event.ydata))
            self.ax[0].plot(event.xdata, event.ydata, "ro")
            self.fig.canvas.draw()

    def onclick_img2(self, event):
        if event.inaxes == self.ax[1] and event.button == 3:
            self.points2.append((event.xdata, event.ydata))
            self.ax[1].plot(event.xdata, event.ydata, "ro")
            self.fig.canvas.draw()

    def run(self):
        self.ax[0].imshow(self.img1)
        self.ax[0].set_title("Image 1")
        self.ax[1].imshow(self.img2)
        self.ax[1].set_title("Image 2")
        plt.show()

    def get_selected_points(self):
        return self.points1, self.points2

    def get_transform_matrix(self, type="affine"):
        if not self.points1 or not self.points2:
            return None
        points1 = np.array(self.points1)
        points2 = np.array(self.points2)
        if type == "affine":
            M = transform.estimate_transform("affine", points2, points1).params
        elif type == "rigid":
            M = transform.estimate_transform(
                "similarity", points2, points1
            ).params
        return M


def image_registration(
    img_fixed,
    img_moving,
    initialtransformation=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
):
    """
    Find transformation between two images using pyStackReg.

    Parameters:
    -----------
    img_fixed : ndarray, shape (x, y)
        2D array representing the fixed image.
    img_moving : ndarray, shape (x, y)
        2D array representing the image which will be transformed into

    Returns:
    --------
    tform : list, shape (6,)
        1x6 list representing the transformation obtained after registration.
        The first four numbers are the rotation matrix.
        The last two numbers are the translation.
    """
    # # Add noise
    # img_moving += np.random.normal(
    #   np.mean(img_moving), np.std(img_moving), np.shape(img_moving)
    # )
    # img_fixed += np.random.normal(
    #   np.mean(img_fixed), np.std(img_fixed), np.shape(img_fixed)
    # )

    # Apply initial transformation
    img_moving_init = image_warp(img_moving, initialtransformation)

    # Registration
    sr = StackReg(StackReg.AFFINE)
    tform = sr.register(img_fixed, img_moving_init)
    tform = np.array(
        [
            tform[0, 0],
            tform[0, 1],
            tform[1, 0],
            tform[1, 1],
            tform[0, 2],
            tform[1, 2],
        ]
    )

    # Multiply transformations.
    tform_fin = (
        tform[:4].reshape(2, 2) * initialtransformation[:4].reshape(2, 2)
    ).flatten()
    tform_fin = np.hstack(
        (
            tform_fin,
            [
                tform[4] + initialtransformation[4],
                tform[5] + initialtransformation[5],
            ],
        )
    )
    return tform_fin


def image_warp(img_moving, tform):
    """
    Transform the input image img_moving according to the geometric
    transformation tform. The center of transformation is the center of
    the image. It is used SimpleITK.

    Parameters:
    -----------

    img_fixed : ndarray, shape (x, y)
        2D array reference image (used by SimpleITK, although the reason is
        not clear).
    tform : list, shape (6,)
        1x6 list representing the geometric transformation to apply.
        The first four numbers are the rotation matrix.
        The last two numbers are the translation.

    Returns:
    --------
    img_tform : ndarray, shape (x, y)
        2D array image after performing the transformation.
        Same size as img_moving.
    tform: ndarray, shape(3,3)
        Matrix containing the geometric transformation applied
        Rotation matrix [0:2,0:2]
        Translation matrix [0,2],[1,2]

    """
    tform = np.array(
        [
            [tform[0], tform[1], tform[4]],
            [tform[2], tform[3], tform[5]],
            [0.0, 0.0, 1.0],
        ]
    )
    img_tform = transform.warp(
        img_moving, transform.AffineTransform(matrix=tform)
    )
    return img_tform


def cosine_stretching(
    mask,
    stack,
    angles,
    crop=True,
    selectPoints=False,
    rot_axis="Y",
    distance_fiducials=0.0,
):
    """
    Obtain the real tilt angles between the images of the stack by image
    registration and apply transformations to correct cosine stretching.
    Transformations are applied to the center of the stack.

    Parameters:
    -----------
    mask : ndarray, shape (z, x, y)
        3D array where the first axis is the number of images. Used to compute
        the transformation matrix.
    stack : ndarray, shape (z, x, y)
        3D array where the first axis is the number of images. Transformations
        obtained are applied to this stack.
    angles : ndarray, shape (z,)
        1D array of nominal tilt angles of the stack. Used to select the image
        reference and an initial transformation matrix.
    crop : bool, optional
        Choose to crop images for registration. Default is True.
    select_points : bool, optional
        Choose if the initial transformation will be obtained by selecting
        points on the images (True), or if it is computed only from the angles
        (False, default).
    rot_axis : str, optional.
        Rotation axis of the images to compute cosine stretching. Options: X, Y.
        Default is Y
    distance_fiducials : float, optional
    metric : str, optional
        Metric to minimize during registration.
        Options: 'meansquares' (default), 'correlation'.
    display : int, optional
        Choose what information to print during registration.
        0 (default, do not display anything),
        1 (display transformation and condition for stopping),
        2 (display transformation, condition for stopping,
        and plot metric during registration).
    niter : int, optional
        Number of iterations for the optimizer gradient descent.
        Default is 1000.

    Returns:
    --------
    mask_ali : ndarray, shape (z, x, y)
        3D array where the first axis is the number of images.
        Transformed input array mask.
    stack_ali : ndarray, shape (z, x, y)
        3D array where the first axis is the number of images.
        Transformed input array stack.
    angles_ali : ndarray, shape (z,)
        1D array of tilt angles of the stack obtained during registration.
    """

    if crop:
        roi = select_roi(
            np.mean(mask, axis=0), step="Select zone for alignment"
        )
        x_ini, y_ini = roi[-2]
        x_fin, y_fin = roi[-1]

        mask_crop = np.zeros_like(mask)
        mask_crop[:, x_ini:x_fin, y_ini:y_fin] = mask[
            :, x_ini:x_fin, y_ini:y_fin
        ]

    else:
        mask_crop = mask

    # Select reference
    idx_ref = np.argmin(np.abs(angles))
    img_fixed = np.float32(mask_crop[idx_ref, :, :])

    # Initialization
    mask_ali = np.zeros_like(mask, dtype=np.float32)
    stack_ali = np.zeros_like(stack, dtype=np.float32)
    angles_ali = np.zeros_like(angles, dtype=np.float32)

    # Loop over angles
    for i_ang in range(0, mask.shape[0]):
        if i_ang == idx_ref:
            mask_ali[i_ang, :, :] = mask[i_ang, :, :]
            stack_ali[i_ang, :, :] = stack[i_ang, :, :]
            angles_ali[i_ang] = 0.0
            print(
                f"{i_ang}: Nominal Angle {angles[i_ang]}, "
                f"Estimated Angle: {angles_ali[i_ang]}"
            )
            continue
        img_moving = np.float32(mask_crop[i_ang, :, :])

        M = None
        if selectPoints:
            # Initial transformation
            M = point_selection(
                mask[i_ang, :, :],
                mask[idx_ref, :, :],
                msg=f"Tilt Angle = {angles[i_ang]}deg",
            )[0]
        if not (M is None):
            M = np.array([M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])
        else:
            M = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        # Perform registration
        tform = image_registration(
            img_fixed,
            img_moving,
            initialtransformation=M,
        )

        # Apply transformations
        mask_ali[i_ang, :, :] = image_warp(mask[i_ang, :, :], tform)

        # Obtain tilt angles.
        tform2ang = lambda a: np.rad2deg(  # noqa: E731
            np.linalg.norm(np.arccos(complex(a, 0)))
        )
        if str(rot_axis) == "Y":  # Cosine stretching applied in X
            angles_ali[i_ang] = tform2ang(tform[0]) * np.sign(angles[i_ang])
        else:  # Cosine stretching applied in Y.
            angles_ali[i_ang] = tform2ang(tform[3]) * np.sign(angles[i_ang])
        print(
            f"{i_ang}: Nominal Angle {angles[i_ang]}, "
            f"Estimated Angle: {angles_ali[i_ang]}"
        )

        # Apply additional translation due to distance between
        # fiducials and magnetic layer in the magnetic images.
        tform = list(tform)
        if str(rot_axis) == "Y":  # Cosine stretching applied in X
            tform[4] += (
                -np.tan(np.deg2rad(angles_ali[i_ang])) * distance_fiducials
            )
        else:  # Cosine stretching applied in Y.
            tform[5] += (
                -np.tan(np.deg2rad(angles_ali[i_ang])) * distance_fiducials
            )
        stack_ali[i_ang, :, :] = image_warp(stack[i_ang, :, :], tform)

    return mask_ali, stack_ali, angles_ali


def join_tilt_series(
    mask,
    stack,
    angle,
    maskRot,
    stackRot,
    angleRot,
    crop=True,
    selectPoints=True,
    initial_angle=90,
):
    """
    Joins two tilt series when the sample is rotated approximately 90 degrees.

    The function estimates the rotation angle by registering the images of both
    tilt series at 0 degrees using SimpleITK. It computes rigid geometric
    transformations and applies them to the rotated tilt series, where the
    center of rotation is the center of the image. It outputs the rotation
    angle.

    Parameters:
    ----------
    mask : ndarray, shape (z, x, y)
        3D array of absorption images or masks stacked along the 0 axis. Used to
        compute the transformation matrix.
    stack : ndarray, shape (z, x, y)
        3D array of magnetic images stacked along the 0 axis. Transformations
        obtained are applied to this stack.
    angles : ndarray, shape (z,)
        1D array of tilt angles of the stack. Used to select the image
        reference.
    mask_rot : ndarray, shape (z, x, y)
        3D array of absorption images or masks stacked along the 0 axis for the
        rotated series. Used to compute the transformation matrix.
    stack_rot : ndarray, shape (z, x, y)
        3D array of magnetic images stacked along the 0 axis for the rotated
        series. Transformations obtained are applied to this stack.
    angles_rot : ndarray, shape (z,)
        1D array of tilt angles of the stack for the rotated series. Used to
        select the image reference.
    crop : bool, optional
        Choose to crop images for registration. Default is True.
    select_points : bool, optional
        Choose if the initial transformation is obtained by selecting points on
        the images (True, default), or computed only from the angles (False).
    initial_angle : float, optional.
        Initial angle (in degrees) for the estimation of the transformation
        for the join tilt series step. Default is 90.

    Returns:
    -------
    mask_all : ndarray, shape (z, x, y)
        3D array of absorption images or masks for both tilt series stacked
        along the 0 axis.
    stack_all : ndarray, shape (z, x, y)
        3D array of magnetic images for both tilt series stacked along
        the 0 axis.
    angles_all : ndarray, shape (z,)
        1D array of tilt angles for both tilt series.
    phi : ndarray, shape (z,)
        1D array of rotation angles for both tilt series.
    """
    # Images at 0deg
    idx_ref_fixed = np.argmin(np.abs(angle))
    img_fixed = np.float32(mask[idx_ref_fixed, :, :])
    idx_ref_moving = np.argmin(np.abs(angleRot))
    img_moving = np.float32(maskRot[idx_ref_moving, :, :])

    if crop:
        roi = select_roi(img_fixed, step="Select zone for alignment")
        x_ini, y_ini = roi[-2]
        x_fin, y_fin = roi[-1]

        m = np.zeros_like(img_fixed)
        m[x_ini:x_fin, y_ini:y_fin] = 1
        img_fixed = img_fixed * m

        plt.imshow(img_fixed)
        plt.title("Reference image")
        plt.show(block=False)
        roi = select_roi(img_moving, step="Select zone for alignment Rot")
        x_ini, y_ini = roi[-2]
        x_fin, y_fin = roi[-1]
        plt.close("all")

        img_moving = img_moving * m
    M = None
    if selectPoints:
        # Initial transformation

        M = point_selection(
            maskRot[idx_ref_moving, :, :],
            mask[idx_ref_fixed, :, :],
            msg="Tilt Angle 0deg",
            type="affine",
        )[0]
    if not (M is None):
        M = np.array([M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])
    else:
        M = np.array(
            [
                np.cos(np.radians(initial_angle)),
                -np.sin(np.radians(initial_angle)),
                np.sin(np.radians(initial_angle)),
                np.cos(np.radians(initial_angle)),
                0,
                0,
            ]
        )

    # Perform registration
    tform = image_registration(
        img_fixed,
        img_moving,
        initialtransformation=M,
    )

    # Initialization
    maskRot_ali = np.zeros_like(maskRot, dtype=np.float32)
    stackRot_ali = np.zeros_like(stackRot, dtype=np.float32)

    if np.abs(tform[0]) < 1:
        phi = np.rad2deg(np.arccos(tform[0]))
    else:
        phi = 0
    print(f"Rotation angle: {phi}")

    # Loop over angles to apply transformations
    for i_ang in range(0, maskRot.shape[0]):
        maskRot_ali[i_ang, :, :] = image_warp(maskRot[i_ang, :, :], tform)
        stackRot_ali[i_ang, :, :] = image_warp(stackRot[i_ang, :, :], tform)

    # Output stacks
    maskAll = np.concatenate((mask, maskRot_ali))
    stackAll = np.concatenate((stack, stackRot_ali))
    phi = np.concatenate(
        (np.zeros(mask.shape[0]), np.multiply(np.ones(maskRot.shape[0]), phi))
    )
    anglesAll = np.concatenate((angle, angleRot))
    return maskAll, stackAll, anglesAll, phi


def magnetic_reconstruction2D(
    xmcd, angles, phi, rot_axis="Y", Lval=1, thickness=1, delta=1
):
    """
    Perform magnetic reconstruction by fitting the Beer-Lambert equation for
    transmission in 2D materials. Reconstruction is fixed to |m| = 1.

    Parameters:
    -----------
    xmcd : ndarray, shape (z, x, y)
        3D array of magnetic signal stack along the 0 axis.
    angles : ndarray, shape (z,)
        1D array of tilt angles of the magnetic images.
    phi : ndarray, shape (z,)
        1D array of rotation angles of the magnetic images.
    Lval : numeric, optional
        Value for attenuation length of magnetic material in meters.
        This value is applied for all the reconstructed volume.
        It can be obtained from the CXRO database. Default: 1
    thickness : numeric, optional
        Value of the thickness of the magnetic material layer for
        reconstruction in meters. Default: 1
    delta : numeric, optional
        Dichroic coefficient. It can be obtained from the energy spectra by
        computing the asymmetry ratio. Default: 1

    Returns:
    --------
    mx : ndarray, shape (x, y)
        Reconstructed magnetization along X axis.
    my : ndarray, shape (x, y)
        Reconstructed magnetization along Y axis.
    mz : ndarray, shape (x, y)
        Reconstructed magnetization along Z axis.
    r2m : ndarray, shape (x, y)
        Value of the coefficient of determination, r², for the reconstruction.
    """

    def cosd(a):
        return np.cos(np.radians(a))

    def sind(a):
        return np.sin(np.radians(a))

    def fun_rotY(m):
        return y - x * (
            sind(angles) * cosd(phi) * m[0]
            + sind(angles) * sind(phi) * m[1]
            + cosd(angles) * m[2]
        )

    def fun_rotX(m):
        return y - x * (
            sind(angles) * cosd(phi) * m[1]
            + sind(angles) * sind(phi) * m[0]
            + cosd(angles) * m[2]
        )

    roi = select_roi(
        np.mean(xmcd, axis=0),
        step="Select zone to crop final stacks for reconstruction",
    )
    x_ini, y_ini = roi[-2]
    x_fin, y_fin = roi[-1]
    xmcd = xmcd[:, x_ini:x_fin, y_ini:y_fin]

    # Initialization
    mx = np.zeros(xmcd.shape[1:])
    mz = np.zeros(xmcd.shape[1:])
    my = np.zeros(xmcd.shape[1:])
    r2m = np.zeros(xmcd.shape[1:])
    angles = angles.flatten()

    for i in range(xmcd.shape[1]):
        for j in range(xmcd.shape[2]):
            x = -(1 / Lval * thickness) / cosd(angles) * delta
            y = xmcd[:, i, j]
            if str(rot_axis) == "Y":
                lsq = least_squares(fun_rotY, [0, 0, 0], method="lm")
            else:
                lsq = least_squares(fun_rotX, [0, 0, 0], method="lm")
            mx[i, j] = lsq.x[0]
            my[i, j] = lsq.x[1]
            mz[i, j] = lsq.x[2]

            # Error
            meas = y
            r2m[i, j] = 1 - np.sum(lsq.fun**2) / (
                np.sum((meas - np.mean(meas)) ** 2)
            )
    return mx, my, mz, r2m


def main(args=None):
    parser = app_parser()
    args = parser.parse_args(args)

    start_time = time.time()

    if args.repeat_step == 0 or args.repeat_step == 1:
        # Load absorption (mask), XMCD and angles from magnetism_xmcd
        with h5py.File(args.stack, "r") as stack_f:
            if "AbsorptionTiltAligned" in stack_f:
                mask = stack_f["AbsorptionTiltAligned"][()]
                signal = stack_f["MagneticSignalTiltAligned"][()]
                angles = stack_f["Angles"][()]
            elif "Absorption2DAligned" in stack_f:
                mask = stack_f["Absorption2DAligned"][()]
                signal = stack_f["MagneticSignal2DAligned"][()]
                angles = stack_f["Angles"][()]
            else:
                print(
                    "ERROR: Dataset named as 'Absorption2DAligned "
                    f"not found in {args.stack}"
                )
        # Compute cosine stretching
        # for normal tomography
        mask_ali, stack_ali, angles_ali = cosine_stretching(
            mask,
            signal,
            angles,
            crop=args.crop_cos,
            selectPoints=args.select_points_cos,
            rot_axis=args.rot_axis,
            distance_fiducials=args.distance_fiducials,
        )

        # Save cosine stretching results
        with h5py.File(
            "cos_stretching.hdf5", "w"
        ) as f:  # TODO: Definition of output dir
            f.create_dataset("Absorption", data=mask_ali, dtype=np.float32)
            f.create_dataset("MagneticSignal", data=stack_ali, dtype=np.float32)
            f.create_dataset("Angles", data=angles_ali, dtype=np.float32)

    if args.repeat_step == 0 or args.repeat_step == 2:
        with h5py.File(args.rotated_stack, "r") as rotated_stack_f:
            if "AbsorptionTiltAligned" in rotated_stack_f:
                rotated_mask = rotated_stack_f["AbsorptionTiltAligned"][()]
                rotated_signal = rotated_stack_f["MagneticSignalTiltAligned"][
                    ()
                ]
                rotated_angles = rotated_stack_f["Angles"][()]
            elif "Absorption2DAligned" in rotated_stack_f:
                rotated_mask = rotated_stack_f["Absorption2DAligned"][()]
                rotated_signal = rotated_stack_f["MagneticSignal2DAligned"][()]
                rotated_angles = rotated_stack_f["Angles"][()]
            else:
                print(
                    "ERROR: Dataset named as 'Absorption2DAligned"
                    f"not found in {args.rotated_stack}"
                )

        # for rotated tomography
        mask_ali_rot, stack_ali_rot, angles_ali_rot = cosine_stretching(
            rotated_mask,
            rotated_signal,
            rotated_angles,
            crop=args.crop_rot_cos,
            selectPoints=args.select_points_rot_cos,
            rot_axis=args.rot_axis,
            distance_fiducials=args.distance_fiducials,
        )

        with h5py.File(
            "cos_stretching_rotated.hdf5", "w"
        ) as f:  # TODO: Definition of output dir
            f.create_dataset("Absorption", data=mask_ali_rot, dtype=np.float32)
            f.create_dataset(
                "MagneticSignal", data=stack_ali_rot, dtype=np.float32
            )
            f.create_dataset("Angles", data=angles_ali_rot, dtype=np.float32)

    if args.repeat_step == 0 or args.repeat_step == 3:
        with h5py.File("cos_stretching.hdf5", "r") as f:
            mask_ali = f["Absorption"][...]
            stack_ali = f["MagneticSignal"][...]
            angles_ali = f["Angles"][...]
        with h5py.File("cos_stretching_rotated.hdf5", "r") as f:
            mask_ali_rot = f["Absorption"][...]
            stack_ali_rot = f["MagneticSignal"][...]
            angles_ali_rot = f["Angles"][...]
        # Join tilt series
        absorption_join, xmcd_join, angles_join, phi_join = join_tilt_series(
            mask_ali,
            stack_ali,
            angles_ali,
            mask_ali_rot,
            stack_ali_rot,
            angles_ali_rot,
            crop=args.crop_join_tilt,
            selectPoints=args.select_points_join_tilt,
            initial_angle=args.initial_angle_join_tilt,
        )
        # Save tilt series results
        with h5py.File(
            "join_tilt_series.hdf5", "w"
        ) as f:  # TODO: Definition of output dir
            f.create_dataset(
                "Absorption", data=absorption_join, dtype=np.float32
            )
            f.create_dataset("MagneticSignal", data=xmcd_join, dtype=np.float32)
            f.create_dataset("Angles", data=angles_join, dtype=np.float32)
            f.create_dataset("Phi", data=phi_join, dtype=np.float32)

    if args.repeat_step == 0 or args.repeat_step == 4:
        with h5py.File(
            "join_tilt_series.hdf5", "r"
        ) as f:  # TODO: Definition of output dir
            xmcd_join = f["MagneticSignal"][...]
            angles_join = f["Angles"][...]
            phi_join = f["Phi"][...]
        # Reconstruction
        mx, my, mz, r2m = magnetic_reconstruction2D(
            xmcd_join,
            angles_join,
            phi_join,
            Lval=args.attenuation_lenght,
            thickness=args.thickness,
            delta=args.dichroic_coefficient,
            rot_axis=args.rot_axis,
        )

        # Show reconstruction
        fig, ax = plt.subplots(2, 2)
        mx_ax = ax[0, 0].imshow(mx)
        ax[0, 0].set_title("mx")
        fig.colorbar(mx_ax, ax=ax[0, 0])
        my_ax = ax[0, 1].imshow(my)
        ax[0, 1].set_title("my")
        fig.colorbar(my_ax, ax=ax[0, 1])
        mz_ax = ax[1, 0].imshow(mz)
        ax[1, 0].set_title("mz")
        fig.colorbar(mz_ax, ax=ax[1, 0])
        r2_ax = ax[1, 1].imshow(r2m)
        ax[1, 1].set_title("r2")
        fig.colorbar(r2_ax, ax=ax[1, 1])
        plt.tight_layout()
        plt.show()

        # Save reconstruction
        with h5py.File(
            "reconstruction.hdf5", "w"
        ) as f:  # TODO: Definition of output directory
            f.create_dataset("mx", data=mx, dtype=np.float32)
            f.create_dataset("my", data=my, dtype=np.float32)
            f.create_dataset("mz", data=mz, dtype=np.float32)
            f.create_dataset("r2m", data=r2m, dtype=np.float32)

    print(
        f"magnetism_2Dreconstruction took {time.time() - start_time} seconds\n"
    )


if __name__ == "__main__":
    main()
