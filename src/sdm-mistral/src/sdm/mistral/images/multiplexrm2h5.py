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
from tinydb.storages import JSONStorage

from sdm.mistral.images import util
from sdm.mistral.parser import get_file_paths
from sdm.mistral.xrm2hdf5 import Xrm2H5Converter


def convert_xrm2h5(xrm_file):
    xrm2h5_converter = Xrm2H5Converter(xrm_file)
    xrm2h5_converter.convert_xrm_to_h5_file()


def multiple_xrm_2_hdf5(
    file_index_db,
    subfolders=False,
    cores=-2,
    update_db=True,
    query=None,
    filter_dic=None,
):
    """Using all cores but one for the computations"""

    start_time = time.time()
    db = TinyDB(file_index_db, storage=CachingMiddleware(JSONStorage))

    if filter_dic is not None:
        file_records = util.filter_db(db, filter_dic)
    # Deprecated
    elif query is not None:
        file_records = db.search(query)
    else:
        file_records = db.all()

    # import pprint
    # printer = pprint.PrettyPrinter(indent=4)
    # printer.pprint(file_records)
    root_path = os.path.dirname(os.path.abspath(file_index_db))
    files = get_file_paths(file_records, root_path, use_subfolders=subfolders)

    # The backend parameter can be either "threading" or "multiprocessing".
    Parallel(n_jobs=cores, backend="multiprocessing")(
        delayed(convert_xrm2h5)(xrm_file) for xrm_file in files
    )

    if update_db:
        util.update_db_func(db, "hdf5_raw", file_records)
    db.close()

    n_files = len(files)
    print(
        "--- Convert from xrm to hdf5 %d files took %s seconds ---\n"
        % (n_files, (time.time() - start_time))
    )
    return db
