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
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import zscore
from tinydb import Query, TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from sdm.mistral.image import Image, mv_projection, store_single_image_in_new_h5
from sdm.mistral.images.util import group_db_records
from sdm.mistral.parser import get_file_paths


def align_and_store_from_fn(
    couple_imgs_to_align_filenames,
    dataset_reference="data",
    dataset_for_aligning="data",
    align_method="cv2.TM_CCOEFF_NORMED",
    roi_size=0.5,
):
    image_ref_fn = couple_imgs_to_align_filenames[0]
    img_ref_obj = Image(
        h5_image_filename=image_ref_fn,
        image_data_set=dataset_reference,
        mode="r",
    )

    image_to_align_fn = couple_imgs_to_align_filenames[1]
    img_to_align_obj = Image(
        h5_image_filename=image_to_align_fn, image_data_set=dataset_for_aligning
    )

    _, mv_vector = img_to_align_obj.align_and_store(
        img_ref_obj, align_method=align_method, roi_size=roi_size
    )
    img_ref_obj.close_h5()
    img_to_align_obj.close_h5()

def align_groups(
    index_filename,
    table="hdf5_proc",
    group_by="zpz",
    dataset_for_aligning="data",
    dataset_reference="data",
    roi_size=0.5,
    align_method="cv2.TM_CCOEFF_NORMED",
    filter_dic={},
    cores=-2,
):
    """Align images of one experiment in 2D.
    One can filter files with matching field:value defining filter_dic.
    The crop of the different images will be done in parallel: all cores
    but one used (Value=-2). Each file, contains a single image to be cropped.
    """

    start_time = time.time()
    root_path = os.path.dirname(os.path.abspath(index_filename))

    if len(filter_dic) > 0:
        print("> Filtering %s" % str(filter_dic.keys()))
    jobs = []
    for datagroup_dic, records in group_db_records(
        index_filename, table, group_by, filter_dic
    ):
        for couple_to_align in _get_couples_to_align(records, root_path):
            jobs.append(
                delayed(align_and_store_from_fn)(
                    couple_to_align,
                    dataset_reference=dataset_reference,
                    dataset_for_aligning=dataset_for_aligning,
                    align_method=align_method,
                    roi_size=roi_size,
                )
            )

    n_groups = len(jobs)
    Parallel(n_jobs=cores, backend="multiprocessing")(jobs)
    print(
        (
            "--- Align %d groups took %s seconds ---\n"
            % (n_groups, (time.time() - start_time))
        )
    )

def _get_couples_to_align(h5_records, root_path):
    files = get_file_paths(h5_records, root_path)
    ref_file = files[0]
    files.pop(0)
    for file in files:
        yield (ref_file, file)

def align_and_store_from_fn_v2(
    couple_imgs_to_align_filenames,
    dataset_reference="data",
    dataset_for_aligning="data",
    align_method="cv2.TM_CCOEFF_NORMED",
    roi_size=0.5,
    subpixel_align=False,
):
    image_ref_fn = couple_imgs_to_align_filenames[0]
    img_ref_obj = Image(
        h5_image_filename=image_ref_fn,
        image_data_set=dataset_reference,
        mode="r",
    )

    image_to_align_fn = couple_imgs_to_align_filenames[1]
    img_to_align_obj = Image(
        h5_image_filename=image_to_align_fn, image_data_set=dataset_for_aligning
    )

    img_to_align_obj.align_and_store_v2(
        img_ref_obj,
        align_method=align_method,
        roi_size=roi_size,
        subpixel_align=subpixel_align,
    )
    img_ref_obj.close_h5()
    img_to_align_obj.close_h5()


def align_groups_v2(
    index_filename,
    table="hdf5_proc",
    group_by="zpz",
    dataset_for_aligning="data",
    dataset_reference="data",
    roi_size=0.5,
    align_method="cv2.TM_CCOEFF_NORMED",
    filter_dic={},
    cores=-2,
    subpixel_align=False,
):
    start_time = time.time()
    root_path = os.path.dirname(os.path.abspath(index_filename))

    if len(filter_dic) > 0:
        print("> Filtering %s" % str(filter_dic.keys()))
    jobs = []
    for datagroup_dic, records in group_db_records(
        index_filename, table, group_by, filter_dic
    ):
        for couple_to_align in _get_couples_to_align(records, root_path):
            jobs.append(
                delayed(align_and_store_from_fn_v2)(
                    couple_to_align,
                    dataset_reference=dataset_reference,
                    dataset_for_aligning=dataset_for_aligning,
                    align_method=align_method,
                    roi_size=roi_size,
                    subpixel_align=subpixel_align,
                )
            )

    n_groups = len(jobs)
    # Parallel(n_jobs=cores, backend="multiprocessing")(jobs)
    with Parallel(n_jobs=10, backend="multiprocessing") as parallel:
        _ = parallel(job for job in jobs)
    print(
        (
            "--- Align %d groups took %s seconds ---\n"
            % (n_groups, (time.time() - start_time))
        )
    )

def align_images(
    file_index_fn,
    table_name="hdf5_proc",
    dataset_for_aligning="data",
    dataset_reference="data",
    roi_size=0.5,
    variable="zpz",
    align_method="cv2.TM_CCOEFF_NORMED",
    date=None,
    sample=None,
    energy=None,
    cores=-2,
    query=None,
    jj=True,
):
    """Align images of one experiment by zpz.
    If date, sample and/or energy are indicated, only the corresponding
    images for the given date, sample and/or energy are cropped.
    The crop of the different images will be done in parallel: all cores
    but one used (Value=-2). Each file, contains a single image to be cropped.
    """

    filter_dic = {}
    if date is not None:
        filter_dic["date"] = date
    if sample is not None:
        filter_dic["sample"] = sample
    if energy is not None:
        filter_dic["energy"] = energy

    align_groups(
        file_index_fn,
        table=table_name,
        group_by=variable,
        dataset_for_aligning=dataset_for_aligning,
        dataset_reference=dataset_reference,
        roi_size=roi_size,
        align_method=align_method,
        filter_dic=filter_dic,
        cores=cores,
    )
