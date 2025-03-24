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
import pprint
import time
from copy import deepcopy
from operator import itemgetter
from typing import Dict, List

import h5py
import numpy as np
from joblib import Parallel, delayed
from tinydb import Query, TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from sdm.mistral.images.util import datagroup_gen, dict2hdf5, filter_db
from sdm.mistral.parser import get_file_paths

FILENAME_FIELDS = {
    "normalized": ["date", "sample", "energy", "zpz"],
    "normalized_simple": ["date", "sample", "position", "energy"],
    "normalized_spectroscopy": ["date", "sample", "position"],
    "normalized_magnetism_many_repetitions": [
        "date",
        "sample",
        # "position",
        "energy",
        "polarisation",
    ],
    "normalized_spectrotomo": ["sample", "energy"],
}
FILENAME_FIELDS["normalized_multifocus"] = FILENAME_FIELDS["normalized_simple"]
FILENAME_FIELDS["aligned"] = FILENAME_FIELDS["normalized"]
FILENAME_FIELDS["aligned_multifocus"] = FILENAME_FIELDS["normalized_simple"]

RECORD_FIELDS = {
    "normalized": ["date", "sample", "energy", "zpz"],
    "normalized_simple": ["date", "sample", "position", "energy"],
    "normalized_spectroscopy": ["date", "sample", "position"],
    "normalized_magnetism_many_repetitions": [
        "date",
        "sample",
        # "position",
        "energy",
        "polarisation",
    ],
    "normalized_spectrotomo": ["sample", "energy"],
}
RECORD_FIELDS["normalized_multifocus"] = RECORD_FIELDS["normalized_simple"]
# RECORD_FIELDS["normalized_magnetism_many_repetitions"] = RECORD_FIELDS[
#    "normalized_simple"
# ]
RECORD_FIELDS["aligned"] = RECORD_FIELDS["normalized"]
RECORD_FIELDS["aligned_multifocus"] = RECORD_FIELDS["normalized_simple"]

STRUCTURE_DICT: Dict[str, Dict[str, Dict[str, List]]] = {
    "normalized": {
        "TomoNormalized": {
            "AverageFF": [],
            "Avg_FF_ExpTime": [],
            "CurrentsFF": [],
            "Currents": [],
            "ExpTimes": [],
            "energy": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
    "normalized_simple": {
        "TomoNormalized": {
            "energy": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
    "normalized_spectroscopy": {
        "SpecNormalized": {
            "Currents": [],
            "ExpTimes": [],
            "energy": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
    "normalized_spectrotomo": {
        "TomoNormalized": {
            "Currents": [],
            "ExpTimes": [],
            "energy": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
    "normalized_magnetism_many_repetitions": {
        "TomoNormalized": {
            "Currents": [],
            "ExpTimes": [],
            "energy": [],
            "polarisation": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
    "aligned": {
        "FastAligned": {
            "energy": [],
            "rotation_angle": [],
            "x_pixel_size": [],
            "y_pixel_size": [],
        }
    },
}

STRUCTURE_DICT["normalized_multifocus"] = STRUCTURE_DICT["normalized"]


def metadata_2_stack_dict(
    hdf5_structure_dict,
    files_for_stack,
    ff_filenames=None,
    type_struct="normalized",
    avg_ff_dataset="data",
):
    """Transfer data from many hdf5 individual image files
    into a single hdf5 stack file.
    This method is quite specific for normalized BL09 images"""

    data_filenames = files_for_stack["data"]

    num_keys = len(hdf5_structure_dict)
    if num_keys == 1:
        k, hdf5_structure_dict = list(hdf5_structure_dict.items())[0]

    def extract_metadata_original(metadata_original, hdf5_structure_dict):
        for dataset_name in hdf5_structure_dict:
            if dataset_name in metadata_original:
                value = metadata_original[dataset_name][()]
                hdf5_structure_dict[dataset_name].append(value)

    if "jj_offset" in files_for_stack:
        jj_offset = files_for_stack["jj_offset"]
        hdf5_structure_dict["jj_offset"] = [jj_offset]

    if "polarisation" in files_for_stack:
        polarisation = files_for_stack["polarisation"]
        hdf5_structure_dict["polarisation"] = [polarisation]

    c = 0
    for file in data_filenames:
        f = h5py.File(file, "r")
        # Process metadata
        metadata_original = f["metadata"]
        extract_metadata_original(metadata_original, hdf5_structure_dict)
        if (
            type_struct == "normalized"
            or type_struct == "normalized_simple"
            or type_struct == "normalized_multifocus"
            or type_struct == "normalized_magnetism_many_repetitions"
            or type_struct == "normalized_spectroscopy"
            or type_struct == "aligned"
            or type_struct == "aligned_multifocus"
            or type_struct == "normalized_spectrotomo"
        ):
            if c == 0:
                pixelsize = metadata_original["pixel_size"][()]
                hdf5_structure_dict["x_pixel_size"].append(pixelsize)
                hdf5_structure_dict["y_pixel_size"].append(pixelsize)
            if (
                "energy" not in hdf5_structure_dict
                and type_struct != "normalized_spectroscopy"
            ):
                hdf5_structure_dict["energy"].append(
                    metadata_original["energy"][()]
                )
            elif (
                "energy" not in hdf5_structure_dict
                and type_struct == "normalized_spectroscopy"
            ):
                hdf5_structure_dict["energy"].append(
                    metadata_original["energy"][()]
                )
            hdf5_structure_dict["rotation_angle"].append(
                metadata_original["angle"][()]
            )
            if (
                "polarisation" not in hdf5_structure_dict
                and type_struct == "normalized_magnetism_many_repetitions"
            ):
                hdf5_structure_dict["polarisation"].append(
                    metadata_original["polarisation"][()]
                )
        if (
            type_struct == "normalized"
            or type_struct == "normalized_spectroscopy"
            or type_struct == "normalized_magnetism_many_repetitions"
            or type_struct == "normalized_spectrotomo"
        ):
            hdf5_structure_dict["ExpTimes"].append(
                metadata_original["exposure_time"][()]
            )
            hdf5_structure_dict["Currents"].append(
                metadata_original["machine_current"][()]
            )
        f.close()
        c += 1

    c = 0
    if ff_filenames and type_struct == "normalized":
        for ff_file in ff_filenames:
            f = h5py.File(ff_file, "r")
            metadata_original = f["metadata"]
            # Process metadata
            if c == 0:
                hdf5_structure_dict["Avg_FF_ExpTime"].append(
                    metadata_original["exposure_time"][()]
                )
                hdf5_structure_dict["AverageFF"] = f[avg_ff_dataset][()]
            hdf5_structure_dict["CurrentsFF"].append(
                metadata_original["machine_current"][()]
            )
            f.close()
            c += 1
    if num_keys == 1:
        hdf5_structure_dict = {k: hdf5_structure_dict}
    return hdf5_structure_dict


def data_2_hdf5(
    h5_stack_file_handler,
    data_filenames,
    ff_filenames=None,
    type_struct="normalized",
    dataset="data",
):
    """Generic method to create an hdf5 stack of images from individual
    images"""

    if (
        type_struct == "normalized"
        or type_struct == "normalized_simple"
        or type_struct == "normalized_multifocus"
        or type_struct == "normalized_magnetism_many_repetitions"
        or type_struct == "normalized_spectrotomo"
    ):
        main_grp = "TomoNormalized"
        main_dataset = "TomoNormalized"
        if ff_filenames and type_struct == "normalized":
            ff_dataset = "FFNormalizedWithCurrent"
    elif type_struct == "normalized_spectroscopy":
        main_grp = "SpecNormalized"
        main_dataset = "spectroscopy_normalized"
    elif type_struct == "aligned" or type_struct == "aligned_multifocus":
        main_grp = "FastAligned"
        main_dataset = "tomo_aligned"
    else:
        pass

    num_img = 0
    for file in data_filenames:
        # Images normalized
        f = h5py.File(file, "r")
        if num_img == 0:
            n_frames = len(data_filenames)
            num_rows, num_columns = np.shape(f[dataset][()])
            h5_stack_file_handler[main_grp].create_dataset(
                main_dataset,
                shape=(n_frames, num_rows, num_columns),
                chunks=(1, num_rows, num_columns),
                dtype="float32",
            )
            h5_stack_file_handler[main_grp][main_dataset].attrs[
                "Number of Frames"
            ] = n_frames
        h5_stack_file_handler[main_grp][main_dataset][num_img] = f[dataset][()]
        f.close()
        num_img += 1

    if ff_filenames and type_struct == "normalized":
        # FF images normalized by machine_current and exp time
        num_img_ff = 0
        for ff_file in ff_filenames:
            f = h5py.File(ff_file, "r")
            if num_img_ff == 0:
                n_ff_frames = len(ff_filenames)
                num_rows, num_columns = np.shape(f[dataset][()])
                h5_stack_file_handler[main_grp].create_dataset(
                    ff_dataset,
                    shape=(n_ff_frames, num_rows, num_columns),
                    chunks=(1, num_rows, num_columns),
                    dtype="float32",
                )
                h5_stack_file_handler[main_grp][ff_dataset].attrs[
                    "Number of Frames"
                ] = n_ff_frames
            h5_stack_file_handler[main_grp][ff_dataset][num_img_ff] = f[
                dataset
            ][()]
            f.close()
            num_img_ff += 1


def make_stack(
    files_for_stack, root_path, type_struct="normalized", suffix="_stack"
):

    if "jj_u" in files_for_stack and "jj_d" in files_for_stack:
        jj_u = files_for_stack["jj_u"]
        jj_d = files_for_stack["jj_d"]
        files_for_stack["jj_offset"] = round((jj_u + jj_d) / 2.0, 2)

    data_files = files_for_stack["data"]
    if "ff" in files_for_stack:
        data_files_ff = files_for_stack["ff"]
    else:
        data_files_ff = None

    # Creation of dictionary
    h5_struct_dict = deepcopy(STRUCTURE_DICT[type_struct])
    data_dict = metadata_2_stack_dict(
        h5_struct_dict,
        files_for_stack,
        ff_filenames=data_files_ff,
        type_struct=type_struct,
    )

    # Creation of hdf5 stack
    values = []
    for field in FILENAME_FIELDS[type_struct]:
        if field in files_for_stack:
            values.append(str(files_for_stack[field]))
    h5_out_fn = "_".join(values) + suffix + ".hdf5"

    record = {}
    for field in RECORD_FIELDS[type_struct]:
        if field in files_for_stack:
            record.update({field: files_for_stack[field]})

    h5_out_fn = os.path.join(root_path, h5_out_fn)
    h5_stack_file_handler = h5py.File(h5_out_fn, "w")
    dict2hdf5(h5_stack_file_handler, data_dict)
    data_2_hdf5(
        h5_stack_file_handler,
        data_files,
        ff_filenames=data_files_ff,
        type_struct=type_struct,
    )

    h5_stack_file_handler.flush()
    h5_stack_file_handler.close()

    record.update(
        {
            "filename": os.path.basename(h5_out_fn),
            "extension": ".hdf5",
            "type": type_struct,
            "stack": True,
        }
    )
    return record


def many_images_to_h5_stack(
    file_index_fn,
    table_name="hdf5_proc",
    type_struct="normalized",
    suffix="_stack",
    date=None,
    sample=None,
    energy=None,
    zpz=None,
    ff=False,
    subfolders=False,
    filter_dic={},
    cores=-2,
):
    """Go from many images hdf5 files to a single stack of images
    hdf5 file.
    Using all cores but one, for the computations"""

    # TODO: spectroscopy normalized not implemented (no Avg FF, etc)
    print("--- Individual images to stacks ---")
    start_time = time.time()
    file_index_db = TinyDB(
        file_index_fn, storage=CachingMiddleware(JSONStorage)
    )
    db = file_index_db
    if table_name is not None:
        file_index_db = file_index_db.table(table_name)

    files_query = Query()

    all_file_records = filter_db(file_index_db, filter_dic)

    root_path = os.path.dirname(os.path.abspath(file_index_fn))
    db.drop_table("hdf5_stacks")
    stack_table = db.table("hdf5_stacks")
    files_list = []

    if type_struct == "normalized" or type_struct == "aligned":
        dates_samples_energies_zpzs = []
        for record in all_file_records:
            dates_samples_energies_zpzs.append(
                (
                    record["date"],
                    record["sample"],
                    record["energy"],
                    record["zpz"],
                )
            )
        dates_samples_energies_zpzs = list(set(dates_samples_energies_zpzs))
        for date_sample_energy_zpz in dates_samples_energies_zpzs:
            date = date_sample_energy_zpz[0]
            sample = date_sample_energy_zpz[1]
            energy = date_sample_energy_zpz[2]
            zpz = date_sample_energy_zpz[3]

            # Query building parts
            da = files_query.date == date
            sa = files_query.sample == sample
            en = files_query.energy == energy
            zp = files_query.zpz == zpz
            ff_false = files_query.FF == False  # noqa: E712
            ff_true = files_query.FF == True  # noqa: E712

            data_files_ff = []
            if file_index_db.search(files_query.FF.exists()):
                # Query command
                query_cmd_ff = da & sa & en & ff_true
                h5_ff_records = file_index_db.search(query_cmd_ff)
                data_files_ff = get_file_paths(
                    h5_ff_records, root_path, use_subfolders=subfolders
                )
            if file_index_db.search(files_query.FF.exists()):
                # Query command
                query_cmd = da & sa & en & zp & ff_false
            else:
                # Query command
                query_cmd = da & sa & en & zp
            h5_records = file_index_db.search(query_cmd)
            h5_records = sorted(h5_records, key=itemgetter("angle"))

            data_files = get_file_paths(
                h5_records, root_path, use_subfolders=subfolders
            )
            files_dict = {
                "data": data_files,
                "ff": data_files_ff,
                "date": date,
                "sample": sample,
                "energy": energy,
                "zpz": zpz,
            }
            files_list.append(files_dict)
    elif (
        type_struct == "normalized_multifocus"
        or type_struct == "normalized_simple"
        or type_struct == "aligned_multifocus"
    ):
        dates_samples_energies = []
        for record in all_file_records:
            dates_samples_energies.append(
                (record["date"], record["sample"], record["energy"])
            )
        dates_samples_energies = list(set(dates_samples_energies))
        for date_sample_energy in dates_samples_energies:
            date = date_sample_energy[0]
            sample = date_sample_energy[1]
            energy = date_sample_energy[2]

            # Query building parts
            da = files_query.date == date
            sa = files_query.sample == sample
            en = files_query.energy == energy

            # Query command
            query_cmd = da & sa & en
            h5_records = file_index_db.search(query_cmd)
            h5_records = sorted(h5_records, key=itemgetter("angle"))

            data_files = get_file_paths(
                h5_records, root_path, use_subfolders=subfolders
            )
            files_dict = {
                "data": data_files,
                "date": date,
                "sample": sample,
                "energy": energy,
            }
            files_list.append(files_dict)

    elif type_struct == "normalized_magnetism_many_repetitions":
        for datagroup_dict in datagroup_gen(all_file_records, "polarisation"):
            records = filter_db(file_index_db, datagroup_dict)
            h5_records = sorted(records, key=itemgetter("angle"))
            data_files = get_file_paths(
                h5_records, root_path, use_subfolders=subfolders
            )
            datagroup_dict["data"] = data_files
            files_list.append(datagroup_dict)
    elif type_struct == "normalized_spectroscopy":
        for datagroup_dict in datagroup_gen(all_file_records, "position"):
            records = filter_db(file_index_db, datagroup_dict)
            data_files = get_file_paths(
                records, root_path, use_subfolders=subfolders
            )
            datagroup_dict["data"] = data_files
            files_list.append(datagroup_dict)
    elif type_struct == "normalized_spectrotomo":
        samples_energies = []
        for record in all_file_records:
            samples_energies.append((record["sample"], record["energy"]))
        samples_energies = list(set(samples_energies))
        for sample_energy in samples_energies:
            sample = sample_energy[0]
            energy = sample_energy[1]

            # Query building parts
            sa = files_query.sample == sample
            en = files_query.energy == energy

            # Query command
            query_cmd = sa & en
            h5_records = file_index_db.search(query_cmd)
            h5_records = sorted(h5_records, key=itemgetter("angle"))

            data_files = get_file_paths(
                h5_records, root_path, use_subfolders=subfolders
            )
            files_dict = {
                "data": data_files,
                "date": date,
                "sample": sample,
                "energy": energy,
            }
            files_list.append(files_dict)

    # Parallelization of making the stacks
    records = Parallel(n_jobs=cores, backend="multiprocessing")(
        delayed(make_stack)(
            files_for_stack, root_path, type_struct=type_struct, suffix=suffix
        )
        for files_for_stack in files_list
    )

    stack_table.insert_multiple(records)
    pretty_printer = pprint.PrettyPrinter(indent=4)
    print("Created stacks:")
    for record in stack_table.all():
        pretty_printer.pprint(record["filename"])
    db.close()

    print(
        (
            "--- Individual images to stacks took %s seconds ---\n"
            % (time.time() - start_time)
        )
    )

