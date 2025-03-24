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

from joblib import Parallel, delayed
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage, MemoryStorage

from sdm.mistral.image import Image, average_images
from sdm.mistral.images.util import filter_db
from sdm.mistral.parser import get_file_paths

AVERAGE_TYPE = {
    "zpz": ["date", "sample", "position", "energy", "angle"],
    "ctbio_zpz": ["date", "sample", "position", "energy", "angle"],
    "repetition": [
        "date",
        "sample",
        "energy",
        "jj_u",
        "jj_d",
        "angle",
        "position",
        "polarisation",
    ],
    "repetition_wo_position": [
        "sample",
        "energy",
        "angle",
        "polarisation",
    ],
    "energy_repetition": ["sample", "polarisation", "angle"],
    "ctbio_repetitions_zpz": [
        "date",
        "sample",
        "energy",
        "angle",
        "position",
        "zpz",
    ],
}

METADATA_UNITS = {
    "energy": "eV",
    "angle": "degree",
    "pixel_size": "um",
    "magnification": "dimensionless",
    "exposure_time": "s",
    "machine_current": "mA",
    "zpz": "um",
}


def average_zpz(img_records):
    num_zpz = len(img_records)
    central_zpz = 0
    for img_record in img_records:
        central_zpz += img_record["zpz"]
    central_zpz /= round(float(num_zpz), 1)
    return round(float(central_zpz), 1)


def average_repetitions(img_records):
    return len(img_records)


AVERAGE_FUNC = {
    "zpz": average_zpz,
    "repetition": average_repetitions,
    "energy_repetition": average_repetitions,
    "repetition_wo_position": average_repetitions,
    "ctbio_repetitions_zpz": average_repetitions,
    "ctbio_zpz": average_repetitions,
}


def average_and_store(
    files_to_average,
    datagroup_dic,
    avg_by,
    avg_value,
    dir_name,
    output_fn,
    dataset_for_average="data",
    dataset_store="data",
    description="",
):

    output_h5_fn = os.path.join(dir_name, output_fn)
    average_images(
        files_to_average,
        dataset_for_average=dataset_for_average,
        description=description,
        store=True,
        output_h5_fn=output_h5_fn,
        dataset_store=dataset_store,
    )

    record = datagroup_dic
    record.update(
        {
            "filename": output_fn,
            "extension": ".hdf5",
            "average": True,
            "avg_by": avg_by,
            avg_by: avg_value,
        }
    )

    img_in_obj = Image(files_to_average[0], mode="r")
    h5_in = img_in_obj.f_h5_handler
    img_avg_obj = Image(output_h5_fn)
    h5_avg = img_avg_obj.f_h5_handler

    metadata_in = "metadata"
    metadata_out = "metadata"
    if metadata_in in h5_in:
        meta_in_grp = h5_in[metadata_in]
        if metadata_out not in h5_avg:
            meta_out_grp = h5_avg.create_group(metadata_out)
        for field, units in METADATA_UNITS.items():
            if "/metadata/%s" % field in h5_in:
                data = meta_in_grp[field][()]
                meta_out_grp.create_dataset(field, data=data)
                meta_out_grp[field].attrs["units"] = units

    h5_in.close()
    img_avg_obj.close_h5()
    return record


def average_image_groups(
    file_index_fn,
    table_name="hdf5_proc",
    dataset_for_average="data",
    avg_by="zpz",
    description="",
    dataset_store="data",
    filter_dic={"FF": True},
    cores=-2,
):
    """Average images of one experiment according to one of the average types.
    One can filter files with matching field:value defining filter_dic.
    The average of the different groups of images will be done in parallel:
    all cores but one used (Value=-2).
    """

    print("> Averaging by %s" % avg_by)

    start_time = time.time()

    root_path = os.path.dirname(os.path.abspath(file_index_fn))

    db = TinyDB(file_index_fn, storage=CachingMiddleware(JSONStorage))

    file_index_db = db.table(table_name)
    if len(filter_dic) > 0:
        print("> Filtering %s" % str(filter_dic.keys()))
        temp_db = TinyDB(storage=MemoryStorage)
        entries = filter_db(db.table(table_name), filter_dic)
        temp_db.insert_multiple(entries)
        file_index_db = temp_db

    all_file_records = file_index_db.all()

    # Create a list of unique datagroups
    datagroup_list = []
    field_list = []
    for record in all_file_records:
        datagroup = []
        for field in AVERAGE_TYPE[avg_by]:
            if field in record:
                if field not in field_list:
                    field_list.append(field)
                datagroup.append(record[field])
        datagroup = tuple(datagroup)
        if datagroup not in datagroup_list:
            datagroup_list.append(datagroup)

    jobs = []
    for datagroup in datagroup_list:
        datagroup_dic = {}
        for i, k in enumerate(field_list):
            datagroup_dic[k] = datagroup[i]

        img_records = filter_db(file_index_db, datagroup_dic)
        avg_value = AVERAGE_FUNC[avg_by](img_records)
        files_to_average = get_file_paths(img_records, root_path)

        output_fn = (
            "_".join(
                list(map(str, datagroup)) + ["avg", str(avg_value), avg_by]
            )
            + ".hdf5"
        )
        fn_first = files_to_average[0]
        dir_name = os.path.dirname(fn_first)

        jobs.append(
            delayed(average_and_store)(
                files_to_average,
                datagroup_dic,
                avg_by,
                avg_value,
                dir_name,
                output_fn,
                dataset_for_average="data",
                dataset_store="data",
                description="",
            )
        )

    n_groups = len(jobs)
    records = Parallel(n_jobs=cores, backend="multiprocessing")(jobs)

    db.drop_table("hdf5_averages")
    averages_table = db.table("hdf5_averages")
    averages_table.insert_multiple(records)

    print(
        "--- Average %d groups, took %s seconds ---\n"
        % (n_groups, (time.time() - start_time))
    )

    # import pprint
    # pobj = pprint.PrettyPrinter(indent=4)
    # print("----")
    # print("average records")
    # for record in records:
    #     pobj.pprint(record)
    db.close()



# Deprecated


def average_image_group_by_energy(
    file_index_fn,
    table_name="hdf5_proc",
    dataset_for_averaging="data",
    variable="repetition",
    description="",
    dataset_store="data",
    date=None,
    sample=None,
    energy=None,
):
    """Method used in energyscan macro. Average by energy.
    Average images by repetition for a single energy.
    If date, sample and/or energy are indicated, only the corresponding
    images for the given date, sample and/or energy are processed.
    All data images of the same energy,
    for the different repetitions are averaged.
    """
    filter_dic = {"FF": False}
    if date is not None:
        filter_dic["date"] = date
    if sample is not None:
        filter_dic["sample"] = sample
    if energy is not None:
        filter_dic["energy"] = energy

    average_image_groups(
        file_index_fn,
        table_name=table_name,
        dataset_for_average=dataset_for_averaging,
        avg_by=variable,
        description=description,
        dataset_store=dataset_store,
        filter_dic=filter_dic,
        cores=-2,
    )


def average_image_group_by_angle(
    file_index_fn,
    table_name="hdf5_proc",
    angle=0.0,
    dataset_for_averaging="data",
    variable="repetition",
    description="",
    dataset_store="data",
    date=None,
    sample=None,
    energy=None,
):
    """Average images by repetition for a single angle.
    If date, sample and/or energy are indicated, only the corresponding
    images for the given date, sample and/or energy are processed.
    All data images of the same angle,
    for the different repetitions are averaged.
    """
    filter_dic = {"FF": False}
    if date is not None:
        filter_dic["date"] = date
    if sample is not None:
        filter_dic["sample"] = sample
    if energy is not None:
        filter_dic["energy"] = energy
    if angle is not None:
        filter_dic["angle"] = angle

    average_image_groups(
        file_index_fn,
        table_name=table_name,
        dataset_for_average=dataset_for_averaging,
        avg_by=variable,
        description=description,
        dataset_store=dataset_store,
        filter_dic=filter_dic,
        cores=-2,
    )
