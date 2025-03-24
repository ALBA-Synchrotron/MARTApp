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


import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
from joblib import Parallel, delayed
from termcolor import colored
from tinydb import Query, TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage, MemoryStorage

from sdm.mistral.image import normalize_ff, normalize_image
from sdm.mistral.images.util import dic_to_query
from sdm.mistral.parser import get_file_paths
from sdm.mistral.util import select_roi


def average_ff(
    file_index_fn,
    table_name="hdf5_proc",
    date=None,
    sample=None,
    energy=None,
    cores=-2,
    query=None,
    ff=True,
    bg=False,
):
    # start_time = time.time()
    file_index_db = TinyDB(
        file_index_fn, storage=CachingMiddleware(JSONStorage)
    )
    if table_name is not None:
        file_index_db = file_index_db.table(table_name)

    files_query = Query()
    if date or sample or energy:
        temp_db = TinyDB(storage=MemoryStorage)
        if date:
            records = file_index_db.search(files_query.date == date)
            temp_db.insert_multiple(records)
        if sample:
            records = temp_db.search(files_query.sample == sample)
            temp_db.drop_tables()
            temp_db.insert_multiple(records)
        if energy:
            records = temp_db.search(files_query.energy == energy)
            temp_db.drop_tables()
            temp_db.insert_multiple(records)
        file_index_db = temp_db

    root_path = os.path.dirname(os.path.abspath(file_index_fn))

    file_records = file_index_db.all()

    # Check if the polarisation is available
    has_polarisation = (
        len(file_index_db.search(files_query.polarisation.exists())) != 0
    )

    # Check if the polarisation is available (if available, include in the
    # conditions)
    has_polarisation = (
        len(file_index_db.search(files_query.polarisation.exists())) != 0
    )

    # Check if there are energies but not angles (e.g. energyscann), and then
    # exluce angles from the analysis and include energies
    include_energy = (
        len(file_index_db.search(files_query.angle.exists())) == 0
        and file_index_db.search(files_query.energy.exists()) != 0
    )

    dates_samples_energies = []
    for record in file_records:
        data = (record["sample"],)
        if has_polarisation:
            data += (record["polarisation"],)
        if include_energy:
            data += (record["energy"],)
        dates_samples_energies.append(data)

    dates_samples_energies = list(set(dates_samples_energies))
    for date_sample_energy in dates_samples_energies:
        # Retrieve basic arguments for searching samples in the DB
        sample = date_sample_energy[0]

        # Construct query to retrieve FFs or BGs
        query_cmd_ff = (
            (files_query.sample == sample)
            & (files_query.FF == ff)  # noqa: E712
            & (files_query.DF == bg)  # noqa: E712
        )
        # Include extra arguments for the search
        if has_polarisation:
            polarisation = date_sample_energy[1]
            query_cmd_ff &= files_query.polarisation == polarisation
        if include_energy:
            energy = date_sample_energy[2 if has_polarisation else 1]
            query_cmd_ff &= files_query.energy == energy

        # Retrieve FF records from the DB and file path, then normalize
        h5_ff_records = file_index_db.search(query_cmd_ff)
        files_ff = get_file_paths(h5_ff_records, root_path)
        normalize_ff(files_ff)


def pool_normalize_image(normalize_arguments):
    return normalize_image(
        image_filename=normalize_arguments["image_filename"],
        average_normalized_ff_img=normalize_arguments[
            "average_normalized_ff_img"
        ],
        average_normalized_df_img=normalize_arguments[
            "average_normalized_df_img"
        ],
        roi_sample=normalize_arguments["roi_sample"],
        roi_ff=normalize_arguments["roi_ff"],
    )


def normalize_images(
    file_index_fn,
    table_name="hdf5_proc",
    date=None,
    sample=None,
    energy=None,
    use_ff=True,
    use_bg=False,
    interpolate_ff=False,
    cores=-2,
    query=None,
    filter_dic={},
    pipeline="",
):
    """Normalize images of one experiment.
    If date, sample and/or energy are indicated, only the corresponding
    images for the given date, sample and/or energy are normalized.
    The normalization of different images will be done in parallel. Each
    file, contains a single image to be normalized.
    .. todo: This method should be divided in two. One should calculate
    the average FF, and the other (normalize_images), should receive
    as input argument, the averaged FF image (or the single FF image).
    """

    start_time = time.time()
    file_index_db = TinyDB(
        file_index_fn, storage=CachingMiddleware(JSONStorage)
    )
    db = file_index_db
    if table_name is not None:
        file_index_db = file_index_db.table(table_name)

    files_query = Query()
    if date or sample or energy:
        temp_db = TinyDB(storage=MemoryStorage)
        if date:
            records = file_index_db.search(files_query.date == date)
            temp_db.insert_multiple(records)
        if sample:
            records = temp_db.search(files_query.sample == sample)
            temp_db.drop_tables()
            temp_db.insert_multiple(records)
        if energy:
            records = temp_db.search(files_query.energy == energy)
            temp_db.drop_tables()
            temp_db.insert_multiple(records)
        file_index_db = temp_db

    root_path = os.path.dirname(os.path.abspath(file_index_fn))

    file_records = file_index_db.all()

    # Check if the polarisation is available (if available, include in the
    # conditions)
    has_polarisation = len(db.search(files_query.polarisation.exists())) != 0

    # Check if there are energies but not angles or only 1 angle (both cases
    # possible e.g. energyscan depending on wheter the txt file is used
    # or not), and then exluce angles from the analysis and include energies
    if pipeline == "spectrotomo":
        include_energy = db.search(files_query.energy.exists()) != 0
    else:
        num_angles = len(
            set(
                [
                    entry["angle"]
                    for entry in db.search(files_query.angle.exists())
                ]
            )
        )
        include_energy = (num_angles == 0 or num_angles == 1) and db.search(
            files_query.energy.exists()
        ) != 0

    dates_samples_energies = []
    for record in file_records:
        data = (record["sample"],)
        if has_polarisation:
            data += (record["polarisation"],)
        if include_energy and not pipeline == "magnetism":
            data += (record["energy"],)
        dates_samples_energies.append(data)

    not_normalized_energies = []
    not_normalized_angles = []

    ff_bg_stacks = {}

    dates_samples_energies = list(set(dates_samples_energies))
    num_files_total = 0
    for date_sample_energy in dates_samples_energies:
        # Retrieve basic arguments for searching samples in the DB
        sample = date_sample_energy[0]

        # Construct query to retrive samples
        query_cmd = (
            (files_query.sample == sample)
            & (files_query.FF == False)  # noqa: E712
            & (files_query.DF == False)  # noqa: E712
        )
        # Include extra arguments for the search
        if has_polarisation:
            polarisation = date_sample_energy[1]
            query_cmd &= files_query.polarisation == polarisation
        if include_energy and not pipeline == "magnetism":
            energy = date_sample_energy[2 if has_polarisation else 1]
            query_cmd &= files_query.energy == energy

        # Include external filtering
        if filter_dic:
            query = dic_to_query(filter_dic)
        if query is not None:
            query_cmd &= query

        # Recover samples entries from the DB
        h5_records = file_index_db.search(query_cmd)

        # Construct query to retrieve FFs
        query_cmd_ff = (
            (files_query.sample == sample)
            & (files_query.FF == True)  # noqa: E712
            & (files_query.DF == False)  # noqa: E712
        )
        # Include extra arguments for the search
        if has_polarisation:
            polarisation = date_sample_energy[1]
            query_cmd_ff &= files_query.polarisation == polarisation
        if include_energy and not pipeline == "magnetism":
            energy = date_sample_energy[2 if has_polarisation else 1]
            query_cmd_ff &= files_query.energy == energy

        # Construct quey to retrive BGs
        query_cmd_df = (
            (files_query.sample == sample)
            & (files_query.FF == False)  # noqa: E712
            & (files_query.DF == True)  # noqa: E712
        )
        # Include extra arguments for the search
        if has_polarisation:
            polarisation = date_sample_energy[1]
            query_cmd_df &= files_query.polarisation == polarisation
        if include_energy and not pipeline == "magnetism":
            energy = date_sample_energy[2 if has_polarisation else 1]
            query_cmd_df &= files_query.energy == energy

        # Retrieve FF and BG entries from the DB
        h5_ff_records = file_index_db.search(query_cmd_ff)
        h5_df_records = file_index_db.search(query_cmd_df)

        # Recover file paths for samples, FFs and BGs
        files_ff = get_file_paths(h5_ff_records, root_path)
        files_df = get_file_paths(h5_df_records, root_path)
        files = get_file_paths(h5_records, root_path)
        n_files = len(files)
        num_files_total += n_files

        # Check if sample file available
        if len(files) <= 0:
            if not include_energy:
                angle = record["angle"]
                not_normalized_angles.append(angle)
            else:
                angle = None
                not_normalized_energies.append(energy)

            element_not_normalized = (
                f"energy {energy}" if include_energy else f"angle {angle}"
            )
            print(
                colored(
                    f"Sample file with {element_not_normalized} "
                    "is not available. "
                    f"{element_not_normalized.capitalize()} is going to be "
                    "removed from the pipeline.",
                    "red",
                )
            )

            continue

        # Four different normalization cases:
        # (0) There are no angles and the normalization is based on the energy
        # (1) Normalize images only by current and exposition time
        #     (not use_ff && not use_bg)
        # (2) Normalize images by current and exposition time, then divide by
        #     averaged FF (use_ff && not use_bg)
        # (3) Normalize images by current and exposition time,
        #     then (img-BG)/(FF-BG) using the closest (in angles)
        #     BG (use_ff && use_bg)
        # The normalization is done using always the same average
        # (for the same date, sample and polarization/energy)
        if has_polarisation:
            ff_bg_stacks[polarisation] = {}
        if use_ff:
            # Case (0)
            if files_ff and include_energy:
                # When there are no angles and only energies, the normalization
                # must be done based on the energy
                avg_norm_ff_files = [normalize_ff(files_ff)]
                # There is only one FF
                closest_angle_per_sample_ff = [0] * n_files
                # There is no BG
                avg_norm_bg = {key: None for key in range(0, n_files)}
            # Cases (2)
            # Obtain normalized averaged FF if there are more than one,
            # and only normalized if there is a single FF
            elif files_ff:
                # Precompute FF averages and normalization
                available_ff_angles_files = {}
                for i in range(0, len(files_ff)):
                    if (
                        h5_ff_records[i]["angle"]
                        in available_ff_angles_files.keys()
                    ):
                        available_ff_angles_files[
                            h5_ff_records[i]["angle"]
                        ] += [files_ff[i]]
                    else:
                        available_ff_angles_files[h5_ff_records[i]["angle"]] = [
                            files_ff[i]
                        ]

                avg_norm_ff_files = {}
                for angle in available_ff_angles_files.keys():
                    avg_norm_ff_files[angle] = normalize_ff(
                        available_ff_angles_files[angle]
                    )
                if has_polarisation:
                    ff_bg_stacks[polarisation]["ff"] = avg_norm_ff_files

                closest_angle_per_sample_ff = {}
                for i in range(0, n_files):
                    sample_angle = h5_records[i]["angle"]

                    min_ff_angle_index = np.abs(
                        np.array(sorted(list(avg_norm_ff_files.keys())))
                        - sample_angle
                    ).argmin()

                    closest_angle_per_sample_ff[i] = sorted(
                        list(avg_norm_ff_files.keys())
                    )[min_ff_angle_index]

                # Look for the closest BG image for each image in files
                avg_norm_bg = {key: None for key in range(0, n_files)}
                if use_bg:
                    # Case (3)
                    if not files_df:
                        raise Exception(
                            "Use of BGs was indicated, but they are not present"
                            " and the image cannot be normalized."
                        )

                    # Accumulate for available BG angle the corresponding
                    # BG file names
                    available_df_angles_files = {}
                    for i in range(0, len(files_df)):
                        if (
                            h5_df_records[i]["angle"]
                            in available_df_angles_files.keys()
                        ):
                            available_df_angles_files[
                                h5_df_records[i]["angle"]
                            ] += [files_df[i]]
                        else:
                            available_df_angles_files[
                                h5_df_records[i]["angle"]
                            ] = [files_df[i]]

                    # Normalize and average BG repetition files per angle
                    avg_norm_bg_files = {}
                    for angle in available_df_angles_files.keys():
                        avg_norm_bg_files[angle] = normalize_ff(
                            available_df_angles_files[angle]
                        )
                    if has_polarisation:
                        ff_bg_stacks[polarisation]["bg"] = avg_norm_bg_files

                    for i in range(0, n_files):
                        sample_angle = h5_records[i]["angle"]

                        angles_list = sorted(list(avg_norm_bg_files.keys()))

                        min_bg_angle_index = np.abs(
                            np.array(angles_list) - sample_angle
                        ).argmin()

                        min_bg_angle = angles_list[min_bg_angle_index]

                        avg_norm_bg[i] = avg_norm_bg_files[min_bg_angle]
            else:
                if not include_energy:
                    angle = record["angle"]
                    not_normalized_angles.append(angle)
                else:
                    angle = None
                    not_normalized_energies.append(energy)
                element_not_normalized = (
                    f"energy {energy}" if include_energy else f"angle {angle}"
                )
                print(
                    colored(
                        "Normalization using FF was indicated, but it has not "
                        f"been possible to normalize {element_not_normalized} "
                        f"(FF with the same angle/energy not available). "
                        f"{element_not_normalized.capitalize()} is going to be "
                        "removed from the pipeline.",
                        "red",
                    )
                )

            # If the interpolarion of FFs is enabled, ROIs for samples and FF
            # must be selected in order to compute factor
            roi_sample = roi_ff = None
            if interpolate_ff:
                with h5py.File(files[int(len(files) / 2)], "r") as sample_file:
                    sample_img = sample_file["data_1"][()]

                ff_img = [*avg_norm_ff_files.values()][
                    int(len(avg_norm_ff_files) / 2)
                ]

                roi_sample = select_roi(
                    image_for_roi_selection=sample_img,
                    step="Select sample ROI for FF interpolation.",
                )
                roi_ff = select_roi(
                    image_for_roi_selection=ff_img,
                    step="Select FF ROI for FF interpolation.",
                )

            # Normalize images in parallel using FFs (and BGs)
            files_to_normalize = [
                {
                    "image_filename": files[k],
                    "average_normalized_ff_img": avg_norm_ff_files[
                        closest_angle_per_sample_ff[k]
                    ],
                    "average_normalized_df_img": avg_norm_bg[k],
                    "roi_sample": roi_sample,
                    "roi_ff": roi_ff,
                }
                for k in range(0, n_files)
            ]

            pool = Pool()
            pool.map(pool_normalize_image, files_to_normalize)
        else:
            # Case (1)
            Parallel(n_jobs=cores, backend="multiprocessing")(
                delayed(normalize_image)(image_filename=h5_file)
                for h5_file in files
            )

    print(
        (
            "--- Normalize %d files took %s seconds ---\n"
            % (num_files_total, (time.time() - start_time))
        )
    )

    db.close()

    return not_normalized_angles, not_normalized_energies, ff_bg_stacks

