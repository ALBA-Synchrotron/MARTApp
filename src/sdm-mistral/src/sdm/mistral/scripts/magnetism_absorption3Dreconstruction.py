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


def reconstruction_parser():
    """
    Defnes parser arguments for the magnetism_absorption2Dreconstruction
    pipeline.
    """
    description = (
        "This pipeline computes the reconstruction of the absorption for "
        f"3D magnetic structures."
    )

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "filename",
        type=str,
        nargs=3,
        help=(
            "HDF5 file and datasets of the tomogram and the angles. "
            "The first dataset must contain sum of the logs of the transmittance images for "
            "different helicities (C+ and C-)."
            "The second dataset is the information of the angles associated to each "
            "projection (in degrees). "
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
            "Logical value 1 (enabled) or 0 (disabled) allowing to "
            "reconstruct for a Continuous Film or an isolated structure. "
            "(Defaults to 0)"
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
            "Integer number indicating the number of simultaneous slices to "
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
            "It will end with _AbsorptionReconstruction.hdf5')"
        ),
        required=False
    )

    parser.add_argument(
        "--only_segmentation",
        help=(
            "Perform only the segmenatation to the dataset selected."
        ),
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--segmentation_mode",
        default=["Threshold"],
        choices=["Threshold","2D_symmetry","Continous_film"],
        nargs=1,
        help=(
            "Choose how to perform segmentation."
        ),
        required=False,
    )

    parser.add_argument(
        "--threshold_val",
        choices=["Otsu", "Triangle", "Huang", "MaxEntropy", "Manual"],
        default=["Otsu"],
        nargs=1,
        help="Value for threshold.",
        required=False,
    )


    parser.add_argument(
        "--mask_negative",
        help=(
            "Mask negative."
        ),
        required=False,
        action='store_true'
    )

    return parser

def volume_threshold(vol,filter_selection='Otsu',negative=False):
    '''
    Function to perform thresholding on a 3D volume using SimpleITK. It includes different filters:
    - Otsu: finds an optimal threshold to separate the input image into foreground and background. Maximizes the variance between these two intensity regions.
    - Triangle: determines a threshold by minimizing the difference between the image histogram and a straight line connecting two points, assuming a peak at the histogram's midpoint between foreground and background intensities.
    - Huang:  finds a threshold based on input image entropy, seeking the threshold that maximizes information gained in separating foreground and background.
    - MaxEntropy: maximizes entropy in the foreground and background regions, computing the threshold that maximizes the sum of entropies in both regions.
    - Manual: the user selects the threshold value from a histogram.
    Parameters:
    -----------
    vol : ndarray, shape (x, y, z)
        3D array containing the volume.
    filter_selection : str, optional
        Specify the thresholding method.
        Options: 'Otsu' (default), 'Triangle','Huang','MaxEntropy'.
    negative : boolean, optional
        It indicates whether to apply thresholding to the complement of the input volume.

    Returns:
    --------
    thresh_value : float
        Threshold value used in the thresholding process.
    thresh_vol : ndarray, shape (x, y, z)
        3D array containing the thresholded volume.
    '''
    vol_sitk = sitk.GetImageFromArray(vol)

    if filter_selection == 'Manual':
        # Choose threshold
        accepted_threshold = False
        thresh_value = 0.0
        thresh_vol = vol
        while not accepted_threshold :
            fig = plt.figure()
            subfig = fig.subfigures(2,1, wspace = 0.7)
            axhist = subfig[0].subplots(1,1)
            axhist.hist(vol.flatten(), bins=75)
            axmask = subfig[1].subplots(1,3)
            axmask[0].imshow(thresh_vol.sum(axis=0))
            axmask[0].set_title("XY")
            axmask[1].imshow(thresh_vol.sum(axis=1))
            axmask[1].set_title("YZ")
            axmask[2].imshow(thresh_vol.sum(axis=2))
            axmask[2].set_title("XZ")
            plt.tight_layout()
            fig.suptitle(
                f" (Current value is {'%.3f' % thresh_value}. "
                "Choose a new one\n or directly click enter to close the window "
                "if the current value is correct.)",
                fontsize=10,
            )
            fig.show()
            new_threshold = fig.ginput(1, timeout=0)
            plt.close()

            if len(new_threshold) == 0:
                accepted_threshold = True
            else:
                thresh_value = new_threshold[0][0]

            if negative:
                thresh_vol_sitk = sitk.BinaryThreshold(vol_sitk, lowerThreshold=thresh_value, upperThreshold=np.max(vol), insideValue=0, outsideValue=1)
            else:
                thresh_vol_sitk = sitk.BinaryThreshold(vol_sitk, lowerThreshold=thresh_value, upperThreshold=np.max(vol), insideValue=1, outsideValue=0)
            thresh_vol = sitk.GetArrayFromImage(thresh_vol_sitk).astype(np.float64)

    else:
        threshold_filters = {'Otsu': sitk.OtsuThresholdImageFilter(),
                             'Triangle' : sitk.TriangleThresholdImageFilter(),
                             'Huang' : sitk.HuangThresholdImageFilter(),
                             'MaxEntropy' : sitk.MaximumEntropyThresholdImageFilter()}
        thresh_filter = threshold_filters[filter_selection]
        if negative:
            thresh_filter.SetInsideValue(0)
            thresh_filter.SetOutsideValue(1)
        else:
            thresh_filter.SetInsideValue(1)
            thresh_filter.SetOutsideValue(0)

        thresh_vol_sitk = thresh_filter.Execute(vol_sitk)
        thresh_value = thresh_filter.GetThreshold()

    print("Threshold used: " + str(thresh_value))   
    thresh_vol = sitk.GetArrayFromImage(thresh_vol_sitk).astype(np.float64)
    return thresh_value, thresh_vol

def segmentation_2Dsymmetry(vol,filter_selection='Otsu',negative='False'):
    '''
    Function to segmentate a volume with 2D symmetry. It first performs a thresholding on a 3D volume using SimpleITK. 
    Then, it asks the user to create a mask using  the XY projection.
    This mask is propagated along the volume where the thresholding  equals 1.
    Parameters:
    -----------
    vol : ndarray, shape (x, y, z)
        3D array containing the volume.
    filter_selection : str, optional
        Specify the thresholding method.
        Options: 'Otsu' (default), 'Triangle','Huang','MaxEntropy'.
    negative : boolean, optional
        It indicates whether to apply thresholding to the complement of the input volume.

    Returns:
    --------
    thresh_value : float
        Threshold value used in the thresholding process.
    thresh_vol : ndarray, shape (x, y, z)
        3D array containing the thresholded volume.
    '''
     # Apply threshold
    print('Segmentation with 2D symmetry')
    val, thresh_vol = volume_threshold(vol,filter_selection=filter_selection,negative=negative)
    m = np.mean(thresh_vol,axis=(1,2))
    img = np.mean(vol,axis = 0)
    img_norm = (img-np.min(img))/(np.max(img)-np.min(img))
    val, img_norm = volume_threshold(np.array([img_norm,np.zeros_like(img)]),filter_selection=filter_selection,negative=negative)
    img_norm = img_norm[0,:,:]
    for i in range(0,np.shape(thresh_vol)[0]):
        if m[i] > 30/(np.shape(thresh_vol)[1]*np.shape(thresh_vol)[2]): # Minimum 50 pixels different than 0.
            thresh_vol[i,:,:] = img_norm
        else:
            thresh_vol[i,:,:] = np.zeros_like(img_norm)
    return thresh_vol

def segmentation_continous_film(vol,filter_selection='Otsu',negative='False'):
    '''
    Function to segmentate a volume with 2D symmetry. It first performs a thresholding on a 3D volume using SimpleITK. 
    Then, it asks the user to create a mask using  the XY projection.
    This mask is propagated along the volume where the thresholding  equals 1.
    Parameters:
    -----------
    vol : ndarray, shape (x, y, z)
        3D array containing the volume.
    filter_selection : str, optional
        Specify the thresholding method.
        Options: 'Otsu' (default), 'Triangle','Huang','MaxEntropy'.
    negative : boolean, optional
        It indicates whether to apply thresholding to the complement of the input volume.

    Returns:
    --------
    thresh_vol : ndarray, shape (x, y, z)
        3D array containing the mask for the magnetic reconstruction.
    mask : ndarray, shape (x, y, z)
        3D array containing the mask for registration.
    '''
    print('Segmentation with continous film')
    thresh_value,thresh_vol = volume_threshold(vol,filter_selection=filter_selection,negative=negative)

    # Get coordinates X,Y of fiducials
    ps = PointSelectionMain(np.mean(vol,axis=0))
    ps.run()
    p1 = ps.get_selected_points()

    vol = np.moveaxis(vol,0,-1) # From Z,X,Y to X,Y,Z
    vol_size = np.shape(vol)
    pz = [vol[np.int16(p[1]),np.int16(p[0]),:].argmax() for p in p1] # Get coordinate Z of points by selecting maximum value in the coordinates X,Y
    coords = np.array([(x-vol_size[0]/2,y-vol_size[1]/2,z-vol_size[2]/2) for (x,y),z in zip(p1,pz)])

    # Ask the user for the thickness 
    thickness, _ = QInputDialog.getText(
            None,
            "Thickness",
            f"\nInsert thickness of the magnetic layer in pixels. ",
        )
    thickness = float(thickness)
    print(f"Thickness = {thickness}")
    # Plane equation (Ax+By+Cz+D=0) with the three points
    normal_vector = np.cross(coords[1] - coords[0], coords[2]- coords[0])
    d = -np.dot(normal_vector, coords[0])
    plane_equation = np.array([normal_vector[0], normal_vector[1], normal_vector[2], d])

    # Create a mask for points below the plane
    [X,Y,Z] = np.mgrid[-vol_size[0]/2:vol_size[0]/2,-vol_size[1]/2:vol_size[1]/2,-vol_size[2]/2:vol_size[2]/2]
    mask = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                distance_to_plane = (np.dot(plane_equation[:3], point) + plane_equation[3]) / np.linalg.norm(plane_equation[:3])
                if distance_to_plane <= thickness and distance_to_plane >= 0:
                    mask[i, j, k] = 1

    mask = np.moveaxis(mask,-1,0) # From X,Y,Z to Z,X,Y
    print("Finished")   
    return thresh_vol,mask

class PointSelectionMain:
    '''
    Function to select points in figure.
    '''
    def __init__(self, img1):
        self.img1 = img1
        self.points1 = []

        self.fig, self.ax = plt.subplots(1, 1)
        self.text_ax = self.fig.add_axes([0.1, 0.9, 0.8, 0.05])
        self.text_ax.set_axis_off()
        self.text_ax.text(
            0.5,
            0.5,
            (
                "Points are selected with the right button on the mouse.\n"
                "When finished, close the window."
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        self.cid1 = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick_img1
        )

    def onclick_img1(self, event):
        if event.inaxes == self.ax and event.button == 3:
            self.points1.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, "ro")
            self.fig.canvas.draw()

    def run(self):
        self.ax.imshow(self.img1)
        self.ax.set_title("Image 1")
        plt.show()

    def get_selected_points(self):
        if not self.points1:
            return None
        img_center = (
            (self.img1.shape[1] - 1) / 2,
            (self.img1.shape[0] - 1) / 2,
        )
        points1 = np.array(self.points1) - img_center
        return self.points1

def launch_reconstruction(args_str):
    start_time = time.time()

    matlab_script_path = pkg_resources.resource_filename(
        "sdm.mistral", "matlab/magnetism_absorption3Dreconstruction"
    )
    cmd = f"{matlab_script_path} " + args_str
    print(cmd)
    os.system(cmd)
    print(
    "magnetism_absorption3Dreconstruction took "
    f"{time.time() - start_time} seconds\n"
)


def main(args=None):
    parser = reconstruction_parser()
    args = parser.parse_args(args)
    if not args.only_segmentation:
        # Reconstruction
        if not args.mod_sxy:
            with h5py.File(args.filename[0],'r') as f:
                s = np.shape(f[args.filename[1]][...])
            args.mod_sxy = s[1:]
        if args.reconstruction_axis[0] == "XTilt": #Matlab rotates images, so the axis has to be rotated
            args.reconstruction_axis[0]= "YTilt" 
        elif args.reconstruction_axis[0] == "YTilt":
            args.reconstruction_axis[0] = "XTilt"
        sorted_arguments = [
            args.filename[0],
            args.filename[2],
            args.filename[1],
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
            args.output_filename[0],
        ]
        # Convert False to 0 and True to 1
        sorted_arguments = [1 if val == True else (0 if val == False else val) for val in sorted_arguments]
        args_str = " ".join(str(arg) for arg in sorted_arguments)

        #print(f"launch reconstruction: {args_str}")
        launch_reconstruction(args_str)     

    # Segmentation
    # The volume is the output of the reconstruction.
    filename = f"{args.output_filename[0]}_AbsorptionReconstruction.hdf5"
    with h5py.File(filename,'r') as f:
        vol = f['Absorption3D'][...]

    if args.segmentation_mode[0] == "Threshold":
        val, vol_thresh = volume_threshold(vol,filter_selection=args.threshold_val[0],negative=args.mask_negative)
        mask = vol_thresh
    elif args.segmentation_mode[0] == "2D_symmetry":
        vol_thresh = segmentation_2Dsymmetry(vol,filter_selection=args.threshold_val[0],negative=args.mask_negative)
        mask = vol_thresh
    elif args.segmentation_mode[0] == "Continous_film":
        vol_thresh, mask = segmentation_continous_film(vol,filter_selection=args.threshold_val[0],negative=args.mask_negative)

    with h5py.File(filename,'a') as f:
        if 'Mask3D' in f:
            del f['Mask3D']
        f.create_dataset('Mask3D',data=mask)
        if 'Mask3DRegistration' in f:
            del f['Mask3DRegistration']
        f.create_dataset('Mask3DRegistration',data=vol_thresh)


if __name__ == "__main__":
    main()