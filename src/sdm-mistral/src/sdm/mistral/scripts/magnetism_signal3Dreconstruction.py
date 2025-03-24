#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2023 ALBA Synchrotron
#
# Authors: Aurelio Hierro Rodriguez, A. Estela Herguedas Alonso,
# Joaquin Gomez Sanchez
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

import argparse
import os
import time
import pkg_resources

import h5py
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QInputDialog
import cv2 as cv


def reconstruction_parser():
    """
    Defnes parser arguments for the magnetism_absorption2Dreconstruction
    pipeline.
    """
    description = (
        "This pipeline computes the reconstruction of the signal for "
        f"3D magnetic structures."
    )

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--filename",
        type=str,
        nargs=3,
        help=(
            "HDF5 file and datasets of the tomogram and the angles. "
            "The first dataset must contain the difference of the logs of the transmittance images for "
            "different helicities (C+ and C-)."
            "The second dataset is the information of the angles associated to each "
            "projection (in degrees). "
        ),
        required=False,
    )

    parser.add_argument(
        "--magnetic_mask",
        type=str,
        nargs=2,
        help=(
            "HDF5 file and datasets of the 3D Mask used for the reconstruction. "
        ),
        required=False,
    )

    parser.add_argument(
        "--registration_mask",
        type=str,
        nargs=2,
        help=(
            "HDF5 file and datasets of the 3D Mask used for registration. "
        ),
        required=False,
    )

    parser.add_argument(
        "--filename_rot",
        type=str,
        nargs=3,
        help=(
            "HDF5 file and datasets of the tomogram and the angles. "
            "The first dataset must contain the difference of the logs of the transmittance images for "
            "different helicities (C+ and C-)."
            "The second dataset is the information of the angles associated to each "
            "projection (in degrees). "
        ),
        required=False
    )

    parser.add_argument(
        "--magnetic_mask_rot",
        type=str,
        nargs=2,
        help=(
            "HDF5 file and datasets of the 3D Mask for the rotated tomogram used for the magnetic reconstruction. "
        ),

    )

    parser.add_argument(
        "--registration_mask_rot",
        type=str,
        nargs=2,
        help=(
            "HDF5 file and datasets of the 3D Mask for the rotated tomogram used for registration. "
        ),

    )
    parser.add_argument(
        "--mod_sxy",
        type=int,
        nargs=2,
        help=(
            "Integers indicating the number of model cells in X and Y direction. "
            "It can be as large as the number of columns and rows in the input tomogram."
            "If not selected it gets the size of the input data"
        ),
        required=False,
    )

    parser.add_argument(
        "--mod_sz",
        type=int,
        nargs=1,
        default=[250],
        help=(
            "Integer indicating the number of model cells in Z direction. "
            "It can be as large as one consider because it is related with the "
            "number of layers to be reconstructed in-depth. (Defaults to 250)"
        ),
        required=False,
    )

    parser.add_argument(
        "--pixel_size",
        type=float,
        nargs=1,
        default=[10],
        help="Number indicating the pixel size in nm. (Defaults to 10nm)",
        required=False,
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        nargs=1,
        default=[20],
        help=(
            "Number of iterations to stop the SIRT algorithm and get the "
            "reconstruction. (Defaults to 20)"
        ),
        required=False,
    )

    parser.add_argument(
        "--lc_flag",
        help=(
            "Logical value allowing to "
            "reconstruct for a Continuous Film or an isolated structure. "
            "(Defaults to False)"
        ),
        required=False,
        action='store_true',
    )

    parser.add_argument(
        "--reconstruction_axis",
        type=str,
        choices=["XTilt", "YTilt"],
        default=["XTilt"],
        nargs=1,
        help="Flag to indicate if the reconstruction is performed by rotating "
        "the X or the Y axis. The first means an stretching in the vertical "
        "direction and the second one in the horizontal direction. "
        " The flag can take 'XTilt' or 'YTilt'. (Defaults to 'XTilt')",
        required=False,
    )

    parser.add_argument(
        "--simult_slcs",
        type=int,
        default=[1],
        nargs=1,
        help=(
            "Integer number indicating the number of simultaneous slides to "
            "be reconstructed. This increases the performance of the "
            "reconstruction if there is enough memory available to speed-up "
            "the reconstruction. Ex: 5 reconstructs grouping the data in "
            "blocks of 5 consecutive slices of the tomogram. (Defaults to 1)"
        ),
        required=False,
    )

    parser.add_argument(
        "--gpu",
        help=(
            "Logical flag to enable the use of the GPU. "
            "It only works if an Nvidia GPU is available with CUDA installed. "
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--save_proj_matrices",
        help=(
            "Logical flag to enable the saving of the projection slices."
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--use_proj_matrices",
        help=(
            "Logical flag to enable the use of pre-computed reconstruction matrices." 
            "These should be stored inside the working directory in a folder called "
            "'Projection_Matrices'. The name of each block of matrices should "
            "be 'Projection_x.mat' with x being the projection number. "
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        nargs=1,
        default=[""],
        help=(
            "Beggining for the output filename with the reconstruction results." 
            "It will end with _MagneticReconstruction.hdf5')"
        ),
        required=False
    )

    parser.add_argument(
        "--output_filename_rot",
        type=str,
        nargs=1,
        default=[""],
        help=(
            "Beggining for the output filename for the rotated data"
             " with the reconstruction results." 
            "It will end with AbsorptionReconstruction.hdf5')"
        ),
        required=False
    )

    parser.add_argument(
        "--only_reconstruction",
        help=(
            "Perform only the reconstruction of the first tomogram."
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--only_registration",
        help=(
            "Perform only the registration and the transformation between volumes."
        ),
        required=False,
        action="store_true",
    )

    return parser

def point_selection(img1, img2, type="rigid"):
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
        Options: 'affine' , 'rigid' (default).

    Returns:
    --------
    M : list, shape (6,)
        1x6 list representing the affine transformation between img1 and img2.
    p1 : ndarray, shape (n, 2)
        2D array containing the points selected in img1.
    p2 : ndarray, shape (n, 2)
        2D array containing the corresponding points selected in img2.
    """
    ps = PointSelectionMain(img1, img2)
    ps.run()
    p1, p2 = ps.get_selected_points()
    M = ps.get_transform_matrix(type=type)
    return M, np.array(p1), np.array(p2)


class PointSelectionMain:
    def __init__(self, img1, img2):
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
                "When finished, close the window."
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

    def get_transform_matrix(self, type="rigid"):
        if not self.points1 or not self.points2:
            return None
        img1_center = (
            (self.img1.shape[1] - 1) / 2,
            (self.img1.shape[0] - 1) / 2,
        )
        img2_center = (
            (self.img2.shape[1] - 1) / 2,
            (self.img2.shape[0] - 1) / 2,
        )
        points1 = np.array(self.points1) - img1_center
        points2 = np.array(self.points2) - img2_center
        if type == "affine":
            M = cv.estimateAffine2D(points2, points1)[0]
        elif type == "rigid":
            M = cv.estimateAffinePartial2D(points2, points1)[0]
        return M

def launch_reconstruction(args_str):
    start_time = time.time()

    matlab_script_path = pkg_resources.resource_filename(
        "sdm.mistral", "matlab/magnetism_signal3Dreconstruction"
    )
    cmd = f"{matlab_script_path} " + args_str
    print(cmd)
    os.system(cmd)

    print(
        "magnetism_signal3Dreconstruction took "
        f"{time.time() - start_time} seconds\n"
    )

def volume_registration(
    vol_fixed,
    vol_moving,
    metric="meansquares",
    initialtransformation=(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
    transformationType="rigid",
    display=0,
    niter=100,
):
    """
    Find transformation between two volumes using SimpleITK.

    Parameters:
    -----------
    vol_fixed : ndarray, shape (x, y, z)
        3D array representing the fixed volume.
    vol_moving : ndarray, shape (x, y, z)
        3D array representing the volume which will be transformed into
        vol_fixed.
    metric : str, optional
        Metric to minimize during registration.
        Options: 'meansquares' (default), 'correlation'.
    initialtransformation: ndarray.
        1D array with the initial transformation to be applied. For affine
        transformations its shape is (12,): the first 9 numbers represent
        the rotation matrix and the last 3 the translation. For rigid
        transformations the shape is (6,): the first 3 numbers represeent
        the angle of rotation around X,Y and Z respectively and the last 3 
        the translation. The rotation matrices are applied as Z*Y*X.

    transformationType : str, optional
        Type of transformation to obtain. Options: 'affine' (default), 'rigid'.
    display : int, optional
        Choose what information to print.
        0 (default, do not display anything),
        1 (display transformation and condition for stopping),
        2 (display transformation, condition for stopping,
        and plot metric during registration).
    niter : int, optional
        Number of iterations for the optimizer gradient descent.
        Default is 100.

    Returns:
    --------
    tform : list, shape (6,)
        1x6 list representing the transformation obtained after registration.
        The first four numbers are the rotation matrix.
        The last two numbers are the translation.
    """
    # Load volumes into sitk
    vol_fixed_itk = sitk.GetImageFromArray(vol_fixed)
    vol_moving_itk = sitk.GetImageFromArray(vol_moving)

    # Initial transformation
    if transformationType == "affine":
        tform_init = sitk.AffineTransform(3)
    elif transformationType == "rigid": 
        tform_init = sitk.Similarity3DTransform()
    tform_init.SetMatrix(initialtransformation[:9])
    tform_init.SetTranslation(initialtransformation[9:])
    tform_init.SetCenter((np.shape(vol_moving)[1]/2-0.5,
                          np.shape(vol_moving)[2]/2-0.5,
                          np.shape(vol_moving)[0]/2-0.5))

    # Initial transformation for moving image
    moving_initial_tform = sitk.CenteredTransformInitializer(
        vol_fixed_itk, vol_moving_itk, sitk.Euler3DTransform(), False
    )

    # Configure the registration setting
    R = sitk.ImageRegistrationMethod()
    R.SetShrinkFactorsPerLevel([8, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetMovingInitialTransform(moving_initial_tform)
    if metric == "meansquares":
        R.SetMetricAsMeanSquares()
    elif metric == "correlation":
        R.SetMetricAsCorrelation()  
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricSamplingPercentage(0.25, seed=42)
    R.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                    numberOfIterations=niter,
    )
    R.SetInitialTransform(tform_init)

    # Perform the registration
    tform_sitk = R.Execute(sitk.Cast(vol_fixed_itk,sitk.sitkFloat32),
                            sitk.Cast(vol_moving_itk,sitk.sitkFloat32))
    if display > 0:
        mat = np.array(initialtransformation[:9]).reshape((3, 3))
        print(f"Initial Rotation {(np.rad2deg(np.arccos((np.trace(mat)-1)/2.0)))}")
        mat = np.array(tform_sitk.GetMatrix()).reshape((3, 3))
        print(f"Rotation {(np.rad2deg(np.arccos((np.trace(mat)-1)/2.0)))}")
        print(
            "Optimizer's stopping condition, "
            f"{R.GetOptimizerStopConditionDescription()}"
        )
    tform = tform_sitk.GetMatrix()+tform_sitk.GetTranslation()
    # tform = tform_sitk.GetParameters()
    print(tform)
    return tform


def volume_warp(vol_moving, vol_fixed, tform):
    """
    Transform the input volume vol_moving according to the geometric
    transformation tform. The center of transformation is the center of
    the volume. It is used SimpleITK.

    Parameters:
    -----------
    vol_moving : ndarray, shape (x, y, z)
        3D array volume to which the transformation is applied.
    vol_fixed : ndarray, shape (x, y, z)
        3D array reference volume (used by SimpleITK, although the reason is
        not clear).
    tform : list, shape (6,)
        1x6 list representing the geometric transformation to apply.
        The first four numbers are the rotation matrix.
        The last two numbers are the translation.

    Returns:
    --------
    vol_tform : ndarray, shape (x, y, z)
        3D array volume after performing the transformation.
        Same size as vol_moving.
    """

    # Load volumes into sitk
    vol_fixed_itk = sitk.GetImageFromArray(vol_fixed)
    vol_moving_itk = sitk.GetImageFromArray(vol_moving)

    # Generate transformation
    tform_itk = sitk.AffineTransform(3)
    tform_itk.SetCenter((np.shape(vol_moving)[1]/2-0.5,
                        np.shape(vol_moving)[2]/2-0.5,
                        np.shape(vol_moving)[0]/2-0.5))
    tform_itk.SetMatrix(tform[:9])
    tform_itk.SetTranslation(tform[9:])

    # Apply transformation
    resample = sitk.ResampleImageFilter()
    resample.SetTransform(tform_itk)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetReferenceImage(vol_fixed_itk)
    resample.SetDefaultPixelValue(0)
    vol_tform_itk = resample.Execute(vol_moving_itk)
    vol_tform_itk = sitk.Cast(vol_tform_itk,vol_moving_itk.GetPixelID())

    # Retrieve volumes into numpy array
    vol_tform = sitk.GetArrayFromImage(vol_tform_itk)
    return vol_tform


def volume_vector_warp(vx, vy, vz, tform):
    """
    Transform the input volume vol_moving according to the geometric
    transformation tform. The center of transformation is the center of
    the volume. It is used SimpleITK.

    Parameters:
    -----------
    vol_moving : ndarray, shape (x, y, z)
        3D array volume to which the transformation is applied.
    vol_fixed : ndarray, shape (x, y, z)
        3D array reference volume (used by SimpleITK, although the reason is
        not clear).
    tform : list, shape (6,)
        1x6 list representing the geometric transformation to apply.
        The first four numbers are the rotation matrix.
        The last two numbers are the translation.

    Returns:
    --------
    vol_tform : ndarray, shape (x, y, z)
        3D array volume after performing the transformation.
        Same size as vol_moving.
    """
    # Apply transformation
    vx = volume_warp(vx, vx, tform)
    vy = volume_warp(vy, vy, tform)
    vz = volume_warp(vz, vz, tform)

    # Rotate vectors
    vx_tform = tform[0]*vx + tform[1]*vy + tform[2]*vz
    vy_tform = tform[3]*vx + tform[4]*vy + tform[5]*vz
    vz_tform = tform[6]*vx + tform[7]*vy + tform[8]*vz

    return vx_tform, vy_tform, vz_tform


def main(args=None):
    parser = reconstruction_parser()
    args = parser.parse_args(args)
    if not args.mod_sxy:
            with h5py.File(args.filename[0],'r') as f:
                s = np.shape(f[args.filename[1]][...])
            args.mod_sxy = s[1:]

    if not args.only_registration:
        # Reconstruction of the first file.
        if args.reconstruction_axis[0] == "XTilt": #Matlab rotates images, so the axis has to be rotated
            args.reconstruction_axis[0]= "YTilt" 
        elif args.reconstruction_axis[0] == "YTilt":
            args.reconstruction_axis[0] = "XTilt"
        sorted_arguments = [
            args.filename[0],
            args.filename[2],
            args.filename[1],
            *args.mod_sxy,
            args.mod_sz[0],
            args.pixel_size[0],
            args.n_iter[0],
            args.lc_flag,
            args.reconstruction_axis[0],
            args.simult_slcs[0],
            args.gpu,
            args.save_proj_matrices,
            args.use_proj_matrices,
            args.output_filename[0],
        ]
        if args.magnetic_mask is not None:
            sorted_arguments += [args.magnetic_mask[0], args.magnetic_mask[1]]

        # Convert False to 0 and True to 1
        sorted_arguments = [1 if val == True else (0 if val == False else val) for val in sorted_arguments]
        args_str = " ".join(str(arg) for arg in sorted_arguments)

        print(f"Magnetic reconstruction: {args_str}")
        launch_reconstruction(args_str)

    if not args.only_reconstruction and not args.only_registration:
        # Reconstruction of the second file
        sorted_arguments = [
            args.filename_rot[0],
            args.filename_rot[2],
            args.filename_rot[1],
            *args.mod_sxy[::-1],
            args.mod_sz[0],
            args.pixel_size[0],
            args.n_iter[0],
            args.lc_flag,
            args.reconstruction_axis[0],
            args.simult_slcs[0],
            args.gpu,
            args.save_proj_matrices,
            args.use_proj_matrices,
            args.output_filename_rot[0],
        ]
        if args.magnetic_mask_rot is not None:
           sorted_arguments += [args.magnetic_mask_rot[0], args.magnetic_mask_rot[1]]

        # Convert False to 0 and True to 1
        sorted_arguments = [1 if val == True else (0 if val == False else val) for val in sorted_arguments]
        args_str = " ".join(str(arg) for arg in sorted_arguments)

        print(f"Magnetic reconstruction: {args_str}")
        launch_reconstruction(args_str)
         

    if not args.only_reconstruction:
        start_time = time.time()

        # Transformation between masks.
        with h5py.File(args.registration_mask[0],'r') as f:
            vol01 = f[args.registration_mask[1]][...]
        
        with h5py.File(args.registration_mask_rot[0],'r') as f:
            vol02 = f[args.registration_mask_rot[1]][...]

        cosd = lambda a: np.cos(np.radians(a))
        sind = lambda a: np.sin(np.radians(a))
        # Change rotation angle
        accepted_angle = False
        angle = 0
        while not accepted_angle :
            t = [0.0, 0.0]
            M = point_selection(np.mean(vol02,axis=0),np.mean(vol01,axis=0),'rigid')[0]
            if not (M is None):
                angle = np.rad2deg(np.nanmean([np.arccos(M[0,0]),np.arccos(M[1,1])]))
                t = [M[0,2], M[1,2]]

            tform_init = ( cosd(angle), -sind(angle), 0.0,
                    sind(angle), cosd(angle), 0.0,
                    0.0, 0.0, 1.0,
                    t[0], t[1], 0.0)

            R_vol = np.multiply([(vol01.shape[1] - 1) / 2, (vol01.shape[2] - 1) / 2,  (vol01.shape[0] - 1) / 2], -1)
            tform = volume_registration(vol01, vol02, transformationType='rigid',display=1,metric='meansquares',
                                                initialtransformation=tform_init)
            vol02_tform = volume_warp(vol02, vol01, tform)
            
            # Create subplots for XY, YZ, and XZ views
            vol_err = (vol01-vol02_tform)**2
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Error: (Mask1-Mask2)²')
            axs[0].imshow(vol_err[:, :, vol_err.shape[2] // 2], cmap='viridis', origin='lower', extent=[0, vol_err.shape[0], 0, vol_err.shape[1]])
            axs[0].set_title('XY View')
            axs[1].imshow(vol_err[vol_err.shape[0] // 2, :, :].T, cmap='viridis', origin='lower', extent=[0, vol_err.shape[2], 0, vol_err.shape[1]])
            axs[1].set_title('YZ View')
            axs[2].imshow(vol_err[:, vol_err.shape[1] // 2, :].T, cmap='viridis', origin='lower', extent=[0, vol_err.shape[0], 0, vol_err.shape[2]])
            axs[2].set_title('XZ View')
            plt.tight_layout()
            plt.show(block=False)
            
            new_angle, _ = QInputDialog.getText(
            None,
            "Initial rotation angle for registration",
            f"Current initial rotation angle is {angle}. If you want to try a new "
            "one, introduce it; if not, leave it blank: ",
            )
            

            if new_angle == '':
                accepted_angle = True
            else:
                angle = float(new_angle)
            plt.close("all")
        with h5py.File("Registration.hdf5", "w") as f:
            f.create_dataset("MaskTS1", data=vol01)
            f.create_dataset("MaskTS2", data=vol02)
            f.create_dataset("MaskTS2Registered", data=vol02_tform)
        print(
        "Volume registration took "
        f"{time.time() - start_time} seconds\n"
        )

        # Apply transformation on magnetic signal.
        with h5py.File(f"{args.output_filename[0]}_MagneticReconstruction.hdf5",'r') as f:
            mx = f["MagneticReconstruction/M1"][...]
            mz = f["MagneticReconstruction/M2"][...]
        with h5py.File(f"{args.output_filename_rot[0]}_MagneticReconstruction.hdf5",'r') as f:
            my_rot = f["MagneticReconstruction/M1"][...]
            mz_rot = f["MagneticReconstruction/M2"][...]

        my_tform, mx_tform, mz_tform = volume_vector_warp(np.zeros_like(my_rot), my_rot, np.zeros_like(mz_rot), tform)

        # Find intensity factor due to rotation
        mat_rot,_ , _ = volume_vector_warp(np.zeros_like(my_rot), np.ones_like(my_rot), np.zeros_like(my_rot), tform)
        f1 = np.abs(np.mean(1/mat_rot[mat_rot!=0]))
        print(f"Intensity Factor = {f1}")

        # Find intensity factor by comparing the signal mz.
        f2 = np.abs(np.mean(mz[np.abs(mz)>0])/np.mean(mz_tform[np.abs(mz_tform)>0]))

        # Apply intensity factors in my
        my = my_tform*f1

        if args.magnetic_mask is not None:
            with h5py.File(args.magnetic_mask[0],'r') as f:
                mask = f[args.magnetic_mask[1]][...]
        else:
            mask = vol01

        with h5py.File("MagneticReconstruction.hdf5", "w") as f:
            f.create_dataset("mx", data=mx)
            f.create_dataset("my", data=my)
            f.create_dataset("mz", data=mz)
            f.create_dataset("Mask3D",data=mask)

        # Normalize vectors
        n = np.sqrt(mx**2+my**2+mz**2)
        n[np.abs(n)<=1e-10] = 1
        # mx /= n
        # my /= n
        # mz /= n
        # Handle NaN and Inf values
        mx = np.nan_to_num(mx, nan=0.0, posinf=0.0, neginf=0.0)
        my = np.nan_to_num(my, nan=0.0, posinf=0.0, neginf=0.0)
        mz = np.nan_to_num(mz, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize the vectors
        n = np.sqrt(mx**2 + my**2 + mz**2)
        mx = np.divide(mx, n, out=np.zeros_like(mx), where=n!=0)
        my = np.divide(my, n, out=np.zeros_like(my), where=n!=0)
        mz = np.divide(mz, n, out=np.zeros_like(mz), where=n!=0)

        with h5py.File("MagneticReconstruction_Norm.hdf5", "w") as f:
            f.create_dataset("mx", data=mx)
            f.create_dataset("my", data=my)
            f.create_dataset("mz", data=mz)
            f.create_dataset("Mask3D",data=mask)
  


if __name__ == "__main__":
    main()
