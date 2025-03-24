#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2025 ALBA Synchrotron
#
# Authors: Joaquin Gomez Sanchez, Gabriel Jover Mañas
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

import shutil
from os import path

import cv2
import h5py
import numpy as np
from pystackreg import StackReg
from scipy.ndimage import shift
from skimage import transform
from skimage.registration import phase_cross_correlation
from xpytools.programs.tiltalign import tilt_align


class Image(object):
    def __init__(
        self, h5_image_filename="default.hdf5", image_data_set="data", mode="r+"
    ):
        self.h5_image_filename = h5_image_filename
        self.f_h5_handler = h5py.File(h5_image_filename, mode)
        self.data_type = np.int32
        self.image = 0
        self.image_dataset = ""
        self.extract_single_image_from_h5(image_data_set)
        self.workflow_step = 1
        self.metadata = None

    def extract_single_image_from_h5(self, data_set="data"):
        image = self.f_h5_handler[data_set][()]
        data_type = type(image[0][0])
        self.data_type = data_type
        self.image = np.array(image, dtype=self.data_type)
        try:
            self.image_dataset = self.f_h5_handler[data_set].attrs["dataset"]
        except Exception:
            self.image_dataset = "unknown_dataset"

    def store_image_in_h5(
        self, image, dataset="default", description="default"
    ):
        """Store a single image in an hdf5 file"""
        precedent_step = int(self.f_h5_handler["data"].attrs["step"])
        self.workflow_step = precedent_step + 1
        if dataset == "default":
            dataset = "data_" + str(self.workflow_step)
        self.f_h5_handler.create_dataset(dataset, data=image)
        self.f_h5_handler[dataset].attrs["step"] = self.workflow_step
        self.f_h5_handler[dataset].attrs["dataset"] = dataset
        self.f_h5_handler[dataset].attrs["description"] = description
        try:
            self.f_h5_handler["data"] = h5py.SoftLink(dataset)
        except Exception:
            del self.f_h5_handler["data"]
            self.f_h5_handler["data"] = h5py.SoftLink(dataset)
        return dataset

    def store_dataset_metadata(
        self,
        dataset="data",
        metadata_dset_name="default",
        metadata_value=None,
        metadata_unit=None,
    ):
        """Store dataset metadata"""
        if metadata_value:
            dset_name = self.f_h5_handler[dataset].attrs["dataset"]
            metadata_grp = "metadata_" + dset_name
            if metadata_grp not in self.f_h5_handler:
                self.metadata = self.f_h5_handler.create_group(metadata_grp)
                self.metadata.create_dataset(
                    metadata_dset_name, data=metadata_value
                )
                if metadata_unit:
                    self.metadata[metadata_dset_name].attrs[
                        "units"
                    ] = metadata_unit

    def normalize_by_constant(
        self,
        constant=None,
        store_normalized_by_constant=False,
        description="Image normalized by a constant",
    ):
        """By default, the constant will be equal to the exposure time
        multiplied by the machine current; otherwise, if the constant is
        indicated, the image is normalized by the indicated value"""
        if not constant:
            exp_time = self.f_h5_handler["metadata"]["exposure_time"][()]
            machine_current = self.f_h5_handler["metadata"]["machine_current"][
                ()
            ]
            constant = exp_time * machine_current
        # If the constant is indicated, the image is divided by it
        img_norm_by_constant = self.image / constant
        if store_normalized_by_constant:
            self.store_image_in_h5(
                img_norm_by_constant, description=description
            )
        return img_norm_by_constant

    def clone_image_dataset(self):
        img = self.image
        try:
            description = self.f_h5_handler[self.image_dataset].attrs[
                "description"
            ]
        except Exception:
            description = "No description available"
        self.store_image_in_h5(img, description=description)

    def crop(self, roi={"top": 26, "bottom": 24, "left": 21, "right": 19}):
        """Crop an image. The roi indicates the pixels to be cut off.
        A default ROI is given to cut
        the image borders"""
        [rows, columns] = np.shape(self.image)
        rows_from = roi["top"]
        rows_to = rows - roi["bottom"]
        columns_from = roi["left"]
        columns_to = columns - roi["right"]
        image_cropped = self.image[rows_from:rows_to, columns_from:columns_to]
        description = "Image " + self.image_dataset + " cropped by " + str(roi)
        return image_cropped, description

    def align_from_file(
        self,
        reference_image_obj,
        align_method="cv2.TM_CCOEFF_NORMED",
        roi_size=0.5,
    ):
        """Align an image taking by reference another image. roi_size
        is entered as input parameter as tant per one of the original
        image size"""
        image_ref = reference_image_obj.image
        image_to_align = self.image
        aligned_image, mv_vector = align(
            image_ref,
            image_to_align,
            align_method=align_method,
            roi_size=roi_size,
        )
        ref_fn = reference_image_obj.h5_image_filename
        ref_dataset_name = reference_image_obj.image_dataset
        description = (
            "Image "
            + self.image_dataset
            + " has been aligned taking as reference image "
            + ref_dataset_name
            + "@"
            + path.basename(ref_fn)
        )
        return aligned_image, mv_vector, description

    def align_and_store(
        self,
        reference_image_obj,
        align_method="cv2.TM_CCOEFF_NORMED",
        roi_size=0.5,
    ):
        aligned_image, mv_vector, description = self.align_from_file(
            reference_image_obj, align_method=align_method, roi_size=roi_size
        )
        new_dataset = self.store_image_in_h5(
            aligned_image, description=description
        )
        self.store_dataset_metadata(
            dataset=new_dataset,
            metadata_dset_name="move_vector",
            metadata_value=mv_vector,
        )
        return aligned_image, mv_vector

    def close_h5(self):
        self.f_h5_handler.flush()
        self.f_h5_handler.close()

    ################################
    #
    # NEW SOLUTIONS
    #
    ################################
    def align_from_file_v2(  # TODO: DELETE
        self,
        reference_image_obj,
        align_method="cv2.TM_CCOEFF_NORMED",
        roi_size=0.5,
    ):
        image_ref = reference_image_obj.image
        image_to_align = self.image

        if align_method == "xpy":
            aligned_image, mv_vector = align_xpy(image_ref, image_to_align)
        elif align_method == "crosscorr-fourier":
            aligned_image, mv_vector = align_subpixel(image_ref, image_to_align)
        elif align_method in ("cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR_NORMED"):
            aligned_image, mv_vector = align(
                image_ref,
                image_to_align,
                align_method=align_method,
                roi_size=roi_size,
            )
        elif align_method == "crosscorr-fourier-subpix":
            aligned_image, mv_vector = align_subpixel(
                image_ref,
                image_to_align,
            )
        else:
            raise Exception(
                f"{align_method} is an invalid alignment alogrithm."
            )

        ref_fn = reference_image_obj.h5_image_filename
        ref_dataset_name = reference_image_obj.image_dataset
        description = (
            "Image "
            + self.image_dataset
            + " has been aligned taking as reference image "
            + ref_dataset_name
            + "@"
            + path.basename(ref_fn)
        )
        return aligned_image, mv_vector, description

    def align_from_file_magnetism(
        self,
        reference_image_obj,
        align_method="cv2.TM_CCOEFF_NORMED",
        roi_size=0.5,
        align_subpixel=False,
    ):
        image_ref = reference_image_obj.image
        image_to_align = self.image

        aligned_image, mv_vector = align_v2(
            image_ref, image_to_align, align_method, align_subpixel
        )

        ref_fn = reference_image_obj.h5_image_filename
        ref_dataset_name = reference_image_obj.image_dataset
        description = (
            "Image "
            + self.image_dataset
            + " has been aligned taking as reference image "
            + ref_dataset_name
            + "@"
            + path.basename(ref_fn)
        )
        return aligned_image, mv_vector, description

    def align_and_store_v2(
        self,
        reference_image_obj,
        align_method="cv2.TM_CCOEFF_NORMED",
        roi_size=0.5,
        subpixel_align=False,
    ):
        aligned_image, mv_vector, description = self.align_from_file_magnetism(
            reference_image_obj,
            align_method=align_method,
            roi_size=roi_size,
            align_subpixel=subpixel_align,
        )
        new_dataset = self.store_image_in_h5(
            aligned_image, description=description
        )
        self.store_dataset_metadata(
            dataset=new_dataset,
            metadata_dset_name="move_vector",
            metadata_value=mv_vector,
        )
        return aligned_image, mv_vector


def copy_h5(input, output):
    """Copy file to a new file"""
    shutil.copy(input, output)


def copy_group(source_group, destination_group):
    """
    Recursively copy all datasets and sub-groups from the source group
    to the destination group.
    """
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            subgroup = destination_group.create_group(key)
            copy_group(item, subgroup)
        elif isinstance(item, h5py.Dataset):
            source_group.copy(item, destination_group, name=key)


def store_single_image_in_new_h5(
    h5_filename, image, description="default", data_set="data", metadata=None
):
    """Store a single image in a new hdf5 file"""
    f = h5py.File(h5_filename, "w")
    data_type = type(image[0][0])
    f.create_dataset(data_set, data=image, dtype=data_type)
    f[data_set].attrs["description"] = description
    if metadata is not None:
        metadata_group = f.create_group("metadata")
        copy_group(metadata["metadata"], metadata_group)
    f.flush()
    f.close()


def store_in_multiple_files(result_image, image_filenames, description=""):
    for image_fn in image_filenames:
        image_obj = Image(h5_image_filename=image_fn)
        image_obj.store_image_in_h5(result_image, description=description)
        image_obj.close_h5()


def add(image_filenames, constant=0, store=False, output_h5_fn="default"):
    """
    Add images (addends),
    A constant can also be added to an image
    """
    description = "Add images and/or add a constant element-wise: \n"
    image_obj = Image(h5_image_filename=image_filenames[0])
    result_image = np.array(image_obj.image, dtype=np.int32)
    dataset = image_obj.image_dataset
    description += dataset + "@" + str(image_obj.h5_image_filename)
    image_obj.close_h5()

    for image_fn in image_filenames[1:]:
        image_obj = Image(h5_image_filename=image_fn)
        dataset = image_obj.image_dataset
        description += (
            " + \n" + dataset + "@" + str(image_obj.h5_image_filename)
        )
        result_image += image_obj.image
        image_obj.close_h5()

    if constant != 0:
        if constant % 1 == 0 and np.issubdtype(result_image[0][0], int):
            constant = int(constant)
        result_image = result_image + constant
        description += " + " + str(constant)

    if store:
        if output_h5_fn == "default":
            store_in_multiple_files(
                result_image, image_filenames, description=description
            )
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                result_image,
                description=description,
                data_set=dataset,
            )
    return result_image


def subtract(image_filenames, constant=0, store=False, output_h5_fn="default"):
    """
    From a reference image (minuend),
    subtract one or more images (subtrahends)
    A constant can also be subtracted to the minuend
    """
    description = (
        "Subtract images and/or subtract a constant "
        "to the minuend image (element-wise): \n"
    )
    image_obj = Image(h5_image_filename=image_filenames[0])
    result_image = np.array(image_obj.image, dtype=np.int32)
    dataset = image_obj.image_dataset
    description += dataset + "@" + str(image_obj.h5_image_filename)
    image_obj.close_h5()

    for image_fn in image_filenames[1:]:
        image_obj = Image(h5_image_filename=image_fn)
        dataset = image_obj.image_dataset
        description += (
            " - \n" + dataset + "@" + str(image_obj.h5_image_filename)
        )
        result_image -= image_obj.image
        image_obj.close_h5()

    if constant != 0:
        if constant % 1 == 0 and np.issubdtype(result_image[0][0], int):
            constant = int(constant)
        result_image = result_image - constant
        description += " - " + str(constant)

    if store:
        if output_h5_fn == "default":
            image_obj = Image(h5_image_filename=image_filenames[0])
            image_obj.store_image_in_h5(result_image, description=description)
            image_obj.close_h5()
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                result_image,
                description=description,
                data_set=dataset,
            )
    return result_image


def multiply(image_filenames, constant=1, store=False, output_h5_fn="default"):
    """
    Multiply images stored in hdf5 files (factors) between them.
    Multiply the resulting image by a constant.
    """
    description = (
        "Multiply images and/or multiply " "element-wise by a constant: \n"
    )
    image_obj = Image(h5_image_filename=image_filenames[0])
    result_image = np.array(image_obj.image, dtype=np.int32)
    dataset = image_obj.image_dataset
    description += dataset + "@" + str(image_obj.h5_image_filename)
    image_obj.close_h5()

    for image_fn in image_filenames[1:]:
        image_obj = Image(h5_image_filename=image_fn)
        dataset = image_obj.image_dataset
        description += (
            " * \n" + dataset + "@" + str(image_obj.h5_image_filename)
        )
        result_image = np.multiply(
            result_image, image_obj.image, casting="same_kind"
        )
        image_obj.close_h5()

    if constant != 1:
        if constant % 1 == 0 and np.issubdtype(result_image[0][0], int):
            constant = int(constant)
        result_image = result_image * constant
        description += " * " + str(constant)

    if store:
        if output_h5_fn == "default":
            store_in_multiple_files(
                result_image, image_filenames, description=description
            )
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                result_image,
                description=description,
                data_set=dataset,
            )
    return result_image


def divide(numerator, denominators, store=False, output_h5_fn="default"):
    """
    Divide a reference image (numerator) stored in a hdf5 file, by one or
    more denominators, which can be images stored in hdf5 files or constants.
    """
    description = (
        "Divide a numerator (image or constant) by other images "
        "and/or constants: \n"
    )
    store_filename = "none"
    try:
        result_image = float(numerator)
        description += str(result_image) + " / ("
    except Exception:
        store_filename = numerator
        image_obj = Image(h5_image_filename=numerator)
        result_image = np.array(image_obj.image, dtype=np.int32)
        dataset = image_obj.image_dataset
        description += dataset + "@" + numerator + " / ("
        image_obj.close_h5()

    for i, denominator in enumerate(denominators):
        try:
            constant = float(denominator)
            result_image = result_image / constant
            if i == 0:
                description += str(constant)
            else:
                description += " * " + str(constant)
        except ValueError:
            image_obj = Image(h5_image_filename=denominator)
            dataset = image_obj.image_dataset
            if i == 0:
                description += dataset + "@" + denominator
                if store_filename == "none":
                    store_filename = denominator
            else:
                description += " * " + dataset + "@" + denominator
            result_image = result_image / image_obj.image
            image_obj.close_h5()
    description += " )"

    if store:
        if output_h5_fn == "default":
            image_obj = Image(h5_image_filename=store_filename)
            image_obj.store_image_in_h5(result_image, description=description)
            image_obj.close_h5()
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                result_image,
                description=description,
                data_set=dataset,
            )
    return result_image


def average_images(
    image_filenames,
    dataset_for_average="data",
    description="",
    store=False,
    output_h5_fn="default",
    dataset_store="data",
):
    """Average images"""
    image_obj = Image(h5_image_filename=image_filenames[0])
    average_image = np.zeros(np.shape(image_obj.image), dtype=type(np.float32))
    num_imgs = len(image_filenames)
    for image_fn in image_filenames:
        image_obj = Image(
            h5_image_filename=image_fn, image_data_set=dataset_for_average
        )
        average_image += image_obj.image
        image_obj.close_h5()
    # Average of images that have been beforehand normalized by a constant
    average_image /= num_imgs
    average_image = np.float32(average_image)
    # Store the average image in the first of the input h5 image file

    if store:
        if output_h5_fn == "default":
            image_obj = Image(h5_image_filename=image_filenames[0])
            description = "Average image: " + description
            image_obj.store_image_in_h5(average_image, description=description)
            image_obj.close_h5()
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                average_image,
                description=description,
                data_set=dataset_store,
            )
    return average_image


def divide_by_constant_and_average_images(
    image_filenames,
    constant=None,
    store_normalized_by_constant=False,
    store=False,
):
    """Normalize each of the image in the list by a constant and average all
    the normalized images.
    If the constant is not indicated, as default, the constant is
    the exposure time multiplied by the machine current. If the images shall
    not be normalized, set the constant to 1."""
    image_obj = Image(h5_image_filename=image_filenames[0])
    image_norm_by_constant = image_obj.normalize_by_constant(constant)
    average_image = np.zeros(
        np.shape(image_obj.image), dtype=type(image_norm_by_constant[0][0])
    )
    image_obj.close_h5()
    num_imgs = len(image_filenames)
    for image_fn in image_filenames:
        image_obj = Image(h5_image_filename=image_fn)
        image_norm_by_constant = image_obj.normalize_by_constant(
            constant, store_normalized_by_constant
        )
        average_image += image_norm_by_constant
        image_obj.close_h5()
    # Average of images that have been beforehand normalized by a constant
    average_image /= num_imgs
    # Store the average image in the first of the input h5 image file
    if store:
        image_obj = Image(h5_image_filename=image_filenames[0])
        description = (
            "Average image calculated after normalizing each "
            "of the input images by a constant. If the constant "
            "is not indicated, its default value is the "
            "multiplication of the exposure time by the "
            "machine current"
        )
        image_obj.store_image_in_h5(average_image, description=description)
        image_obj.close_h5()
    return average_image


def get_normalized_ff(ff_img_filenames):
    if isinstance(ff_img_filenames, list):
        ff_img_obj = Image(h5_image_filename=ff_img_filenames[0])
    else:
        ff_img_obj = Image(h5_image_filename=ff_img_filenames)

    return ff_img_obj.image


def normalize_ff(ff_img_filenames):
    if isinstance(ff_img_filenames, list):
        ff_img_obj = Image(h5_image_filename=ff_img_filenames[0])
    else:
        ff_img_obj = Image(h5_image_filename=ff_img_filenames)

    if isinstance(ff_img_filenames, list) and len(ff_img_filenames) > 1:
        ff_img_obj.close_h5()
        # Average of FF images that are beforehand normalized by its
        # corresponding exposure times and machine currents
        ff_norm_image = divide_by_constant_and_average_images(
            ff_img_filenames, store_normalized_by_constant=True, store=True
        )
    else:
        # Normalize FF image by exposure_time and machine_current
        ff_norm_image = ff_img_obj.normalize_by_constant()

    return ff_norm_image


def normalize_image(
    image_filename,
    ff_img_filenames=None,
    df_img_filenames=None,
    average_normalized_ff_img=None,
    average_normalized_df_img=None,
    roi_sample=None,
    roi_ff=None,
    store_normalized=True,
    output_h5_fn="default",
):
    """
    Normalize BL09 hdf5 image: Normalize image by current, exposure time,
    and FlatField (FF) image (in case ff_img_filenames is a single file), or
    by average FF images (in case ff_img_filenames is a list of FF filenames).
    Each FF image is, beforehand, normalized by its corresponding
    current and exposure time.
    :param image_filename: the hdf5 image filename to be normalized
    :param ff_img_filenames: hdf5 FF image filename(s)
    :param average_normalized_ff_img: average normalized FF image. If defined,
    the FF filenames are not used.
    :param store_normalized: (Bool) True if normalized image has to be stored
    :return: normalized image
    """

    image_obj = Image(h5_image_filename=image_filename)

    # Check FF structure and normalize
    ff_norm_image = None
    if average_normalized_ff_img is not None:
        ff_norm_image = average_normalized_ff_img
    elif ff_img_filenames is not None:
        if isinstance(ff_img_filenames, list):
            # TODO: Check if the input as list is happening somewhere
            ff_img_obj = Image(h5_image_filename=ff_img_filenames[0])
        else:
            ff_img_obj = Image(h5_image_filename=ff_img_filenames)

        if np.shape(image_obj.image) != np.shape(ff_img_obj.image):
            raise Exception(
                "Image dimensions does not correspond with "
                "ff image dimensions"
            )
        ff_norm_image = normalize_ff(ff_img_filenames)

    # Check DF structure and normalize (if available)
    df_norm_image = None
    if average_normalized_df_img is not None:
        df_norm_image = average_normalized_df_img
    elif df_img_filenames is not None:
        if isinstance(df_img_filenames, list):
            df_img_obj = Image(h5_image_filename=df_img_filenames[0])
        else:
            df_img_obj = Image(h5_image_filename=df_img_filenames)

        if np.shape(image_obj.image) != np.shape(df_img_obj.image):
            raise Exception(
                "Image dimensions does not correspond with "
                "df image dimensions"
            )
        df_norm_image = normalize_ff(df_img_filenames)

    # Normalize main image by exposure_time and machine_current
    img_norm_by_constant = image_obj.normalize_by_constant()

    if roi_sample is None or roi_ff is None:
        factor = 1.0
    else:
        s_x0, s_y0 = roi_sample[0]
        s_x1, s_y1 = roi_sample[1]
        ff_x0, ff_y0 = roi_ff[0]
        ff_x1, ff_y1 = roi_ff[1]
        factor = np.mean(img_norm_by_constant[s_x0:s_x1, s_y0:s_y1]) / np.mean(
            ff_norm_image[ff_x0:ff_x1, ff_y0:ff_y1]
        )

    if ff_norm_image is None:
        # Normalized image only by (average) FF, taking into account
        # exposure times and machine currents
        normalized_image = img_norm_by_constant
    elif df_norm_image is None:
        normalized_image = img_norm_by_constant / (ff_norm_image * factor)
    else:
        normalized_image = (img_norm_by_constant - df_norm_image) / (
            (ff_norm_image * factor) - df_norm_image
        )

    # Store the resulting normalized image in the main image h5 file
    if store_normalized:
        dataset = image_obj.image_dataset
        description = dataset + "@" + path.basename(image_filename)
        if average_normalized_ff_img is not None or (
            isinstance(ff_img_filenames, list) and len(ff_img_filenames) > 1
        ):
            description += (
                " normalized by average FF, using exposure time "
                "and machine current. To calculate the average "
                "FF, each FF image has been, beforehand, "
                "normalized by its exposure time and "
                "machine current"
            )
        else:
            description += (
                " has been normalized by single FF, "
                "using its corresponding exposure times and "
                "machine currents"
            )

        if output_h5_fn == "default":
            image_obj.store_image_in_h5(
                normalized_image, description=description
            )
        else:
            store_single_image_in_new_h5(
                output_h5_fn,
                normalized_image,
                description=description,
                data_set=dataset,
            )

    image_obj.close_h5()
    return normalized_image, ff_norm_image


def roi_parameters_selection(image, roi_size=0.5):
    """Compute ROI parameters. 'size' is given on 'tant per one' of
    the input image."""

    numrows, numcols = np.shape(image)
    # Take half of the pixels and round to the next closest even number
    width_template = numrows * roi_size
    width_template = int(np.ceil(width_template / 2.0) * 2)
    height_template = numcols * roi_size
    height_template = int(np.ceil(height_template / 2.0) * 2)

    # central image pixel
    central_pixel_rows = int(numrows / 2)
    central_pixel_cols = int(numcols / 2)

    row_template_from = central_pixel_rows - height_template / 2
    col_template_from = central_pixel_cols - width_template / 2

    roi_parameters = [
        row_template_from,
        col_template_from,
        height_template,
        width_template,
    ]
    return roi_parameters


def find_mv_vector(coords_base_roi, coords_mv_roi):
    """How much to move a projection"""

    # coords_base_roi being the base template coordinates
    # (the one that will not move)
    coords_base_roi = np.array(coords_base_roi)
    coords_mv_roi = np.array(coords_mv_roi)

    # The vector should bring the image to be aligned (mv_image)
    # to the base image (fixed image)
    mv_vector = coords_base_roi - coords_mv_roi
    # print(mv_vector)
    return mv_vector


def mv_projection(image, mv_vector, fill_val=0):
    """Move a given image by a certain amount"""
    rows = image.shape[0]
    cols = image.shape[1]
    mv_vector = (int(mv_vector[0]), int(mv_vector[1]))
    mvr = abs(mv_vector[0])
    mvc = abs(mv_vector[1])
    ei = np.ones((rows, cols), dtype="float32") * fill_val
    pt = image

    if mv_vector[0] == 0 and mv_vector[1] == 0:
        ei[:, :] = pt[:, :]

    elif mv_vector[0] > 0 and mv_vector[1] == 0:
        ei[mvr:rows, :] = pt[0 : rows - mvr, :]

    elif mv_vector[0] < 0 and mv_vector[1] == 0:
        ei[0 : rows - mvr, :] = pt[mvr:rows, :]

    elif mv_vector[0] == 0 and mv_vector[1] > 0:
        ei[:, mvc:cols] = pt[:, 0 : cols - mvc]

    elif mv_vector[0] == 0 and mv_vector[1] < 0:
        ei[:, 0 : cols - mvc] = pt[:, mvc:cols]

    elif mv_vector[0] > 0 and mv_vector[1] > 0:
        ei[mvr:rows, mvc:cols] = pt[0 : rows - mvr, 0 : cols - mvc]

    elif mv_vector[0] > 0 and mv_vector[1] < 0:
        ei[mvr:rows, 0 : cols - mvc] = pt[0 : rows - mvr, mvc:cols]

    elif mv_vector[0] < 0 and mv_vector[1] > 0:
        ei[0 : rows - mvr, mvc:cols] = pt[mvr:rows, 0 : cols - mvc]

    elif mv_vector[0] < 0 and mv_vector[1] < 0:
        ei[0 : rows - mvr, 0 : cols - mvc] = pt[mvr:rows, mvc:cols]

    moved_image = ei
    return moved_image


def mv_projection_subpixel(image, mv_vector, fill_val=0):
    """Move a given image by a certain amount"""
    tform = np.array(
        [[1.0, 0.0, -mv_vector[1]], [0.0, 1.0, -mv_vector[0]], [0.0, 0.0, 1.0]]
    )

    tform_skimage = transform.AffineTransform(matrix=tform)
    moved_image = transform.warp(np.array(image), tform_skimage)

    return moved_image


def align_in_fourier(image_ref, image_to_align, roi_size=0.5):
    template = image_ref

    if isinstance(template[0][0], (np.floating, float)):
        if type(template[0][0]) != np.float32:
            template = template.astype(np.float32)
    else:
        template = template.astype(np.uint8)

    if isinstance(image_to_align[0][0], (np.floating, float)):
        if type(image_to_align[0][0]) != np.float32:
            image_to_align = image_to_align.astype(np.float32)
    else:
        image_to_align = image_to_align.astype(np.uint8)

    mv_vector, _, _ = phase_cross_correlation(
        reference_image=template,
        moving_image=image_to_align,
        space="real",
    )

    # Move the projection thanks to the found move vector
    aligned_image = mv_projection(image_to_align, list(map(int, mv_vector)))

    return aligned_image, mv_vector


def align_subpixel(image_ref, image_to_align):
    template = image_ref

    if isinstance(template[0][0], (np.floating, float)):
        if type(template[0][0]) != np.float32:
            template = template.astype(np.float32)
    else:
        template = template.astype(np.uint8)

    if isinstance(image_to_align[0][0], (np.floating, float)):
        if type(image_to_align[0][0]) != np.float32:
            image_to_align = image_to_align.astype(np.float32)
    else:
        image_to_align = image_to_align.astype(np.uint8)

    shifted, _, _ = phase_cross_correlation(
        reference_image=template,
        moving_image=image_to_align,
        space="real",
        upsample_factor=500,
    )

    aligned_image = shift(
        image_to_align, shift=(shifted[0], shifted[1]), mode="constant"
    )

    return aligned_image, shifted


def align_xpy(image_ref, image_to_align):
    aligned_image, mv_vector = tilt_align(
        stk=np.array([image_ref, image_to_align]).astype(np.float32),
        tiltList=None,
        refId=0,
    )  # [1]

    return aligned_image[1], np.array(
        [mv_vector[1][1][-1], mv_vector[1][0][-1]]
    )


def align_v2(image_ref, image_to_align, method, subpixel):
    # Determine mv_projection method
    if subpixel:
        mv_projection_m = mv_projection_subpixel
    else:
        mv_projection_m = mv_projection

    # Align
    if (
        method == "cv2.TM_CCOEFF_NORMED"
        or method == "corrcoeff"
        or method == "cv2.TM_CCORR_NORMED"
        or method == "crosscorr"
    ):
        # Prepare template
        roi_size = 0.5
        roi_parameters = roi_parameters_selection(image_ref, roi_size)
        row_tem_from = int(roi_parameters[0])
        col_tem_from = int(roi_parameters[1])
        h = int(roi_parameters[2])
        w = int(roi_parameters[3])

        template = image_ref[
            row_tem_from : row_tem_from + h, col_tem_from : col_tem_from + w
        ]

        # Correct types if needed
        if isinstance(template[0][0], (np.floating, float)):
            if type(template[0][0]) != np.float32:
                template = template.astype(np.float32)
        else:
            template = template.astype(np.uint8)

        if isinstance(image_to_align[0][0], (np.floating, float)):
            if type(image_to_align[0][0]) != np.float32:
                image_to_align = image_to_align.astype(np.float32)
        else:
            image_to_align = image_to_align.astype(np.uint8)

        # Resolve alignment methods
        if method == "cv2.TM_CCOEFF_NORMED" or method == "corrcoeff":
            align_method = "cv2.TM_CCOEFF_NORMED"
        elif method == "cv2.TM_CCORR_NORMED" or method == "crosscorr":
            align_method = "cv2.TM_CCORR_NORMED"

        # Apply template Matching from cv2
        result = cv2.matchTemplate(image_to_align, template, eval(align_method))
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

        # In openCV first indicate the columns and then the rows.
        top_left_base = (col_tem_from, row_tem_from)

        # Apply template Matching from cv2
        result = cv2.matchTemplate(image_to_align, template, eval(align_method))
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

        # In openCV first indicate the columns and then the rows.
        top_left_base = (col_tem_from, row_tem_from)
        top_left_move = (max_loc[0], max_loc[1])

        # Finde mv_vector
        mv_vector = find_mv_vector(top_left_base, top_left_move)
        rows = mv_vector[1]
        cols = mv_vector[0]
        mv_vector = (rows, cols)

        aligned_image = mv_projection_m(image_to_align, mv_vector)

        return aligned_image, mv_vector
    elif method == "crosscorr-fourier":
        # Correct type if needed (required by phase_cross_correlation)
        if isinstance(image_ref[0][0], (np.floating, float)):
            if type(image_ref[0][0]) != np.float32:
                image_ref = image_ref.astype(np.float32)
        else:
            image_ref = image_ref.astype(np.uint8)

        if isinstance(image_to_align[0][0], (np.floating, float)):
            if type(image_to_align[0][0]) != np.float32:
                image_to_align = image_to_align.astype(np.float32)
        else:
            image_to_align = image_to_align.astype(np.uint8)

        # Compute shift
        mv_vector, _, _ = phase_cross_correlation(
            reference_image=image_ref,
            moving_image=image_to_align,
            space="real",
            upsample_factor=500
            if subpixel
            else 1,  # TODO:Determine upsample factor
        )

        # Align image
        aligned_image = mv_projection_m(
            image=image_to_align, mv_vector=mv_vector, fill_val=1
        )

        return aligned_image, tuple(mv_vector)
    elif method == "pyStackReg":
        sr = StackReg(StackReg.TRANSLATION)

        tform = sr.register(image_ref, image_to_align)
        mv_vector = [(0, 0), (-(tform[1, 2]), -(tform[0, 2]))]

        aligned_image = mv_projection_m(
            image=image_to_align, mv_vector=mv_vector[1], fill_val=1
        )

        return aligned_image, mv_vector[1]
    elif method == "xpy" or method == "of":
        _, mv_vector = tilt_align(
            stk=np.array([image_ref, image_to_align]).astype(np.float32),
            tiltList=None,
            refId=0,
        )
        mv_vector = (mv_vector[1][1][-1], mv_vector[1][0][-1])

        aligned_image = mv_projection_m(
            image=image_to_align, mv_vector=mv_vector, fill_val=1
        )

        return aligned_image, mv_vector
    else:
        raise Exception(f"{method} is an invalid alignment alogrithm.")


def align(
    image_ref, image_to_align, align_method="cv2.TM_CCOEFF_NORMED", roi_size=0.5
):
    """Align an image taking by reference another image. roi_size
    is entered as input parameter as tant per one of the original
    image size"""
    roi_parameters = roi_parameters_selection(image_ref, roi_size)
    row_tem_from = int(roi_parameters[0])
    col_tem_from = int(roi_parameters[1])
    h = int(roi_parameters[2])
    w = int(roi_parameters[3])

    template = image_ref[
        row_tem_from : row_tem_from + h, col_tem_from : col_tem_from + w
    ]
    # template = np.array(template)

    # template matching from cv2 only works with float 32, or
    # with uint8 (from 0 to 256)
    if isinstance(template[0][0], (np.floating, float)):
        if type(template[0][0]) != np.float32:
            template = template.astype(np.float32)
    else:
        template = template.astype(np.uint8)

    if isinstance(image_to_align[0][0], (np.floating, float)):
        if type(image_to_align[0][0]) != np.float32:
            image_to_align = image_to_align.astype(np.float32)
    else:
        image_to_align = image_to_align.astype(np.uint8)

    # Apply template Matching from cv2
    result = cv2.matchTemplate(image_to_align, template, eval(align_method))
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

    # In openCV first indicate the columns and then the rows.
    top_left_base = (col_tem_from, row_tem_from)

    # If you are using cv2.TM_SQDIFF as comparison method,
    # minimum value gives the best match.
    if align_method in ["cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]:
        top_left_move = (min_loc[0], min_loc[1])
    else:
        top_left_move = (max_loc[0], max_loc[1])

    mv_vector = find_mv_vector(top_left_base, top_left_move)
    rows = mv_vector[1]
    cols = mv_vector[0]
    mv_vector = (rows, cols)

    # Move the projection thanks to the found move vector
    aligned_image = mv_projection(image_to_align, mv_vector)

    return aligned_image, mv_vector
