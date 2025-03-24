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

import argparse
import os
import time
from collections import defaultdict
from os import listdir, remove
from os.path import isfile, join
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QInputDialog
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from sdm.mistral.images.imagestostack import many_images_to_h5_stack
from sdm.mistral.images.multiplealign import align_groups_v2
from sdm.mistral.images.multipleaverage import (
    average_image_group_by_angle,
    average_image_groups,
)
from sdm.mistral.images.multiplecrop import crop_images
from sdm.mistral.images.multiplenormalization import (
    average_ff,
    normalize_images,
)
from sdm.mistral.images.multiplexrm2h5 import multiple_xrm_2_hdf5
from sdm.mistral.images.util import copy2proc_multiple, dic_to_query
from sdm.mistral.scripts.parser import (
    get_db,
    parse_crop,
    parse_get_db,
    parse_stack,
    parse_table,
    str2bool,
)
from sdm.mistral.util import remove_all_from_db, save_ff_bg_stacks


WORKFLOW = "xrm2hdf5 >> crop >> normalize >> align2D >> average >> [stack]"


def app_parser():
    """
    Application parser
    :return: parser for magnetism workflow
    """
    description = (
        "magnetism: many repetition images at different angles."
        " Normally using 2 different polarizations.\n\n%s" % WORKFLOW
    )

    parser = argparse.ArgumentParser(
        parents=[parse_get_db, parse_crop, parse_table, parse_stack],
        description=description,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--energy", action="store_true")

    parser.add_argument(
        "--ff",
        action="store_true",
        help="- If present: Create a stack of FF images normalized"
        "by exposition time and current. "
        "It cannot be combined with other pipeline options.",
    )

    parser.add_argument(
        "--bg",
        action="store_true",
        help="- If present: Create a stack of BG images normalized"
        "by exposition time and current. "
        "It cannot be combined with other pipeline options.",
    )

    parser.add_argument(
        "--norm_ff",
        type=str2bool,
        default="True",
        help="Normalize using FF (dafault: true).",
    )

    parser.add_argument(
        "--sub_bg",
        type=str2bool,
        default="True",
        help="Substract (normalize) using BG (default true)",
    )

    parser.add_argument(
        "--th",
        type=float,
        default=None,
        help=("Angle theta to pre-process data" + " referred to this angle"),
    )

    parser.add_argument(
        "--interpolate_ff",
        type=str2bool,
        default="False",
        help="If set, missing FF will be interpolated.",
    )

    parser.add_argument(
        "--align_method",
        choices=[
            "crosscorr",
            "corrcoeff",
            "crosscorr-fourier",
            "of",
            "pyStackReg",
        ],
        type=str,
        default="crosscorr",
        help="Method used for the 2D alignment between repetitions.",
        required=False,
    )

    parser.add_argument(
        "--subpixel_align",
        type=str2bool,
        default="True",
        help="If set, 2D alignment will be done using subpixel accuracy.",
    )

    parser.add_argument(
        "--zscore_threshold",
        type=float,
        default=None,
        help=(
            "Threshold for repetition outliers exclusion. If not indicated, "
            "outliers are not going to be deleted."
        ),
    )

    parser.add_argument(
        "--delete_proc_files",
        type=str2bool,
        default="True",
        help="If set to true (default), processing files are deleted.",
    )

    parser.add_argument(
        "--save_normalization_stacks",
        type=str2bool,
        default="True",
        help=(
            "If combined with --stack it saves FF and BG stacks inside "
            "the final HDF5 file (default)."
        ),
    )

    parser.add_argument(
        "--delete_prev_exec",
        type=str2bool,
        default="false",
        help=("If present the HDF5 files and the JSON DB file will be deleted"),
    )

    return parser


def plot_outliers(
    filenames, intensities, grad_int_norm, outliers_idx, threshold, FF
):
    _, ax1 = plt.subplots(figsize=(20, 10))
    x = range(len(filenames))

    # Add intensity curve
    ax1.plot(x, intensities, "b-", label="Intensity")
    ax1.set_xlabel("Files")
    ax1.set_ylabel("Intensity", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Add gradient curve
    ax2 = ax1.twinx()
    ax2.plot(x, grad_int_norm, "g-", label="Gradient")
    ax2.set_ylabel("Gradient", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    # Draw gradient curve in red when grad > threshold
    for idx in outliers_idx:
        if idx > 0:
            ax2.plot(
                x[idx - 1 : idx + 1], grad_int_norm[idx - 1 : idx + 1], "r-"
            )
        else:
            ax2.plot(x[idx : idx + 2], grad_int_norm[idx : idx + 2], "r-")

    # Set marks for the outliers in the x-axis
    ax1.set_xticks(outliers_idx)
    ax1.set_xticklabels([""] * len(outliers_idx), rotation=90, ha="center")

    # Define the threshold in the plot
    ax2.axhline(y=threshold, linestyle="--", color="gray")

    # Adjust the plot view
    plt.subplots_adjust(right=0.85)
    ax1.set_xlim(-1, len(filenames))

    # Add a list of files to delete
    text_str = "\n".join(
        map(lambda s: s[:-10], np.array(filenames)[outliers_idx])
    )
    text_str = r"$\bf{Files\ to\ delete:}$" + "\n" + text_str
    plt.gcf().text(
        0.9,
        0.5,
        text_str,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Add plot title and show
    plt.title(
        f"Intensities and intensity gradients for {'FF' if FF else 'samples'}"
    )

    plt.show(block=False)


def delete_outlier_repetitions(
    index_filename: str = "index.json", threshold: float = 0.5, FF: bool = False
):
    """Delete outlier repetitions with abs(z-score) >= threshold.

    Parameters
    ----------
    index_filename : str, optional
        DB filename, by default "index.json".
    threshold : float, optional
        Threshold for the gradient, by default 3.0.
    FF: bool, optional
        If true, the DB table hdf5_proc will be filtered by FF.
    """
    # Prepare DB and processing table
    db = TinyDB(index_filename, storage=CachingMiddleware(JSONStorage))
    table_hdf5_proc = db.table("hdf5_proc")

    # Filter flat-fields and brackgrounds
    samples_query = dic_to_query({"FF": FF, "DF": False})
    sample_entries = table_hdf5_proc.search(samples_query)

    # Recover projection images and group by angle_pol
    anglepol_filenames = defaultdict(list)
    for entry in sample_entries:
        anglepol_filenames[(entry["angle"], entry["polarisation"])].append(
            entry["filename"]
        )

    # Compute intensities and gradients
    filenames: List[str] = []
    angle_pol: List[tuple] = []
    intensities: List[float] = []
    gradient_intensities: List[float] = []
    for anglepol, files in anglepol_filenames.items():
        # Sort filenames (to sort repetitions)
        files = sorted(files)
        # Accumulate filenames
        filenames = filenames + files
        # Accumulate angle_pol
        angle_pol = angle_pol + [anglepol] * len(files)

        # Compute and accumulate intensities
        intensity_files = []
        for file in files:
            with h5py.File(Path(file).resolve(), "r") as f:
                last_data = (
                    "/data_2" if "/data_2" in f.keys() else "/data_1"
                )  # TODO: DOUBLE CHECK
                intensity_files.append(np.mean(f[last_data][()]))
        intensities = intensities + intensity_files

        # Compute and accumulate gradients
        gradients = np.gradient(intensity_files)
        gradients_norm = np.abs(gradients / np.linalg.norm(gradients))
        gradient_intensities = gradient_intensities + gradients_norm.tolist()

    # Sort all the filenames, intensities and gradient intensities by angle_pol
    sorted_indices = sorted(
        range(len(angle_pol)), key=lambda k: (angle_pol[k][1], angle_pol[k][0])
    )
    filenames = [filenames[i] for i in sorted_indices]
    intensities = [intensities[i] for i in sorted_indices]
    gradient_intensities = [gradient_intensities[i] for i in sorted_indices]

    # Decide thresholding (dinamically)
    accepted_threshold = False
    while not accepted_threshold:
        # Thresholding
        thresholding = np.array(gradient_intensities) > threshold
        idx_outliers = np.where(thresholding)[0]

        # Improve outliers detection taking into account neighbors
        for i in range(len(idx_outliers) - 1):
            dist = np.abs(idx_outliers[i + 1] - idx_outliers[i])
            if dist < 5:
                idx_outliers = np.concatenate(
                    (
                        idx_outliers,
                        np.arange(idx_outliers[i] + 1, idx_outliers[i + 1]),
                    )
                )
        idx_outliers = np.sort(np.unique(idx_outliers))

        # Show which ones are going to be deleted
        plot_outliers(
            filenames,
            intensities,
            gradient_intensities,
            idx_outliers,
            threshold,
            FF,
        )

        # Choose new threshold
        threshold, _ = QInputDialog.getText(
            None,
            f"Gradient thresholding for {'FFs' if FF else 'samples'}",
            f"Current threshold is {threshold}. If you want to try a "
            "new one, introduce it; if not, leave it blank:",
        )
        if threshold == "":
            accepted_threshold = True
        else:
            threshold = float(threshold)

        plt.close("all")
    plt.close("all")

    db.close()


def partial_preprocesing(
    db_filename,
    group_by,
    crop,
    filter_dic={},
    is_ff=False,
    is_bg=False,
    use_ff=True,
    use_bg=True,
    interpolate_ff=False,
    zscore_threshold=None,
    align_method=None,
    subpixel_align=False,
):
    # Check if HDF5 files already in database
    with TinyDB(db_filename, storage=CachingMiddleware(JSONStorage)) as db:
        hdf5_raw_table_exists = "hdf5_raw" in db.tables()

    # Multiple xrm 2 hdf5 files: working with many single images files
    if not hdf5_raw_table_exists:
        multiple_xrm_2_hdf5(db_filename, filter_dic=filter_dic)

    # Copy of multiple hdf5 raw data files to files for processing
    copy2proc_multiple(
        db_filename, filter_dic=filter_dic, purge=True, magnetism_partial=True
    )

    # Multiple files hdf5 images crop: working with single images files
    if crop:
        crop_images(db_filename, filter_dic=filter_dic)

    # Delete outlier repetitions
    if zscore_threshold is not None:
        delete_outlier_repetitions(
            index_filename=db_filename, threshold=zscore_threshold, FF=False
        )
        delete_outlier_repetitions(
            index_filename=db_filename, threshold=zscore_threshold, FF=True
        )

    # Normalize multiple hdf5 files: working with many single images files
    if is_ff:
        average_ff(db_filename, ff=True, bg=False)
    elif is_bg:
        average_ff(db_filename, ff=False, bg=True)
    else:
        not_norm_angles, not_norm_energies, ff_bg_stacks = normalize_images(
            db_filename,
            filter_dic=filter_dic,
            use_ff=use_ff,
            use_bg=use_bg,
            interpolate_ff=interpolate_ff,
            pipeline="magnetism",
        )

        # Clean database
        if len(not_norm_angles) > 0:
            remove_all_from_db(db_filename, "angle", not_norm_angles)
        if len(not_norm_energies) > 0:
            remove_all_from_db(db_filename, "energy", not_norm_energies)

    filter_dic.update({"FF": is_ff, "DF": is_bg})
    # Align multiple hdf5 files: working with many single images files
    # align_groups(db_filename, group_by=group_by, filter_dic=filter_dic)
    align_groups_v2(
        db_filename,
        group_by=group_by,
        filter_dic=filter_dic,
        align_method=align_method,
        subpixel_align=subpixel_align,
    )

    return db_filename, ff_bg_stacks


def main(args=None):
    """
    - Convert from xrm to hdf5 individual image hdf5 files
    - Copy raw hdf5 to new files for
      processingtxm2nexuslib/workflows/magnetism.py:54
    - Crop borders
    - Normalize
    - Create stacks by date, sample, and energy,
      with multiple angles in each stack
    """
    parser = app_parser()
    args = parser.parse_args(args)

    # Delete previous execution if requested
    if args.delete_prev_exec:
        os.system(
            "rm -rf -- *energy_repetitionenergy_repetition* *proc* *FFnorm* *.json"
        )

    db_filename = get_db(args)

    print(WORKFLOW)

    start_time = time.time()

    # Align and average by repetition
    group_by = "energy_repetition"

    # Dictionary for database filtering
    filter_dic = {}

    # Save if only FF or BG should be processed
    if args.ff:
        filter_dic["FF"] = True
    if args.bg:
        filter_dic["DF"] = True

    # Indicate if a specific angle should be take into account
    if args.th:
        filter_dic["angle"] = args.th

    # Partial preprocessing includes:
    # (1) Conversion from xrm to HDF5 at single image level
    # (2) Copy of images to processing table inside DB
    # (3) Image cropping by ROI
    # (4.1) If FF: Call average_ff, which normalizes by current
    #       and exposition time, and averages the FF if many.
    # (4.2) If BG: Call average_ff, which normalizes by current
    # (4.3) Normalize sample images with FF/BG/nothing (current and exposition).
    _, ff_bg_stacks = partial_preprocesing(
        db_filename,
        group_by,
        args.crop,
        filter_dic=filter_dic,
        is_ff=args.ff,
        is_bg=args.bg,
        use_ff=args.norm_ff,
        use_bg=args.sub_bg,
        interpolate_ff=args.interpolate_ff,
        zscore_threshold=args.zscore_threshold,
        align_method=args.align_method,
        subpixel_align=args.subpixel_align,
    )

    # Average multiple hdf5 files:
    # working with many single images files
    if args.th is None:
        average_image_groups(
            db_filename, avg_by=group_by, filter_dic=filter_dic
        )
    else:
        average_image_group_by_angle(
            db_filename, variable=group_by, angle=args.th
        )

    if args.stack:
        # Build up hdf5 stacks from individual images
        # Stack of variable angle. Each of the images has been done by
        # averaging many repetitions of the image at the same energy,
        # angle... The number of repetitions by each of the images in this
        # stack files could be variable.

        suffix = "_"
        if args.ff:
            suffix += "FF"
        elif args.bg:
            suffix += "BG"
        else:
            if args.norm_ff:
                suffix += "FFnorm"
            else:
                suffix += "norm"
            if args.sub_bg:
                suffix += "BGsub"
        suffix += "_stack"

        many_images_to_h5_stack(
            db_filename,
            table_name=args.table_for_stack,
            type_struct="normalized_magnetism_many_repetitions",
            suffix=suffix,
        )

        # Create stacks for FFs and BGs
        if args.save_normalization_stacks:
            save_ff_bg_stacks("./", ff_bg_stacks)

    # Delete 'proc' fils to save space
    if args.delete_proc_files:
        proc_files = [
            join(".", f)
            for f in listdir(".")
            if isfile(join(".", f))
            and any(f.endswith(end) for end in [".hdf5", ".h5"])
            and "proc" in f
        ]
        for f in proc_files:
            remove(f)

    print(
        "magnetism preprocessing took %d seconds\n" % (time.time() - start_time)
    )


if __name__ == "__main__":
    main()

WORKFLOW = "xrm2hdf5 >> crop >> normalize >> align2D >> average >> [stack]"


