#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2022 ALBA Synchrotron
#
# Authors: Joaquin Gomez Sanchez, A. Estela Herguedas Alonso
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
import datetime
import glob
import os
import time
from argparse import RawTextHelpFormatter

import h5py
from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage

from sdm.mistral.xrmnex import XradiaFile

INDEX_FILENAME = "index.json"
PARAM_AXIS_MAP = {
    "energy": "Energy",
    "angle": "Sample Theta",
    "zpz": "Zone Plate Z",
    "jj_u": "jj_u",
    "jj_d": "jj_d",
    "polarisation": "polarisation",
    "x_position": "Sample X",
    "y_position": "Sample Y",
}

FIELD_TYPE = {
    "date": str,
    "sample": str,
    "position": int,
    "energy": float,
    "angle": float,
    "zpz": float,
    "polarisation": int,
    "repetition": int,
}


def read_file_parameters(filename):
    parameters = {}

    basename = os.path.basename(filename)
    parameters["sample"] = basename.split("_")[1]
    parameters["FF"] = "_FF" in basename
    parameters["DF"] = "_DF" in basename or "_BG" in basename

    if filename.endswith(".xrm"):
        with XradiaFile(filename) as xrm_file:
            try:
                date_time_txt = xrm_file.get_single_date()
                date_time = datetime.datetime.fromisoformat(date_time_txt)
                parameters["date"] = date_time.strftime("%Y%m%d")
            except RuntimeError:
                print(
                    "acquisition date and time could not be found in file ",
                    filename,
                )
            # parameters['repetition'] =
            for k, v in PARAM_AXIS_MAP.items():
                if v in xrm_file.axes_names:
                    parameters[k] = round(xrm_file.get_position(v), 1)
    elif filename.endswith((".hdf5", ".h5")):
        with h5py.File(filename, "r") as h5_file:
            try:
                date_time = datetime.datetime.fromisoformat(
                    h5_file["/metadata/date_time_acquisition"][()].decode(
                        "utf-8"
                    )
                )
                parameters["date"] = date_time.strftime("%Y%m%d")
            except RuntimeError:
                print(
                    "acquisition date and time could not be found in file ",
                    filename,
                )

            for k, v in PARAM_AXIS_MAP.items():
                try:
                    parameters[k] = round(h5_file[f"/metadata/{k}"][()], 1)
                except:  # noqa: E722
                    continue
    else:
        raise Exception(f"Unknown file type: {filename}")

    if parameters["x_position"] and parameters["y_position"]:
        parameters["position"] = "%.1fx%.1fy" % (
            parameters["x_position"],
            parameters["y_position"],
        )

    return parameters


def parse_file_parameters(filename, pattern):
    basename = os.path.basename(filename)
    root, extension = os.path.splitext(basename)
    tags = root.split("_")
    keys = pattern.split("_")

    parameters = {}
    parameters["FF"] = False
    parameters["DF"] = False
    if "FF" in tags:
        parameters["FF"] = True
        tags.remove("FF")
    if "DF" in tags:
        parameters["DF"] = True
        tags.remove("DF")
    if "BG" in tags:
        parameters["DF"] = True
        tags.remove("BG")
    for key, tag in zip(keys, tags):
        if key == "ignore":
            continue
        elif key in FIELD_TYPE:
            parameters[key] = FIELD_TYPE[key](tag)
        else:
            print(f"Warning: unknown key {key}")
            parameters[key] = str(tag)

    return parameters


def create_db(db_filename, extension=".xrm", pattern=None, extra_param=[]):
    start_time = time.time()

    files = glob.glob("*%s" % extension, recursive=False)
    # files += glob.glob("*/*%s" % extension, recursive=False)

    samples = []
    for i, filename in enumerate(files):
        folder = os.path.dirname(filename)
        basename = os.path.basename(filename)
        extension = os.path.splitext(filename)[1]
        parameters = {}
        parameters["processed"] = False
        parameters["subfolder"] = folder
        parameters["filename"] = basename
        parameters["extension"] = extension
        parameters.update(read_file_parameters(filename))
        if pattern:
            file_params = parse_file_parameters(filename, pattern)
            parameters.update(file_params)
        for ep in extra_param:
            k, v = ep.split("=")
            parameters[k] = float(v)
        samples.append(parameters)

    n_files = len(samples)
    db_path = os.path.abspath(db_filename)
    with TinyDB(db_path, storage=CachingMiddleware(JSONStorage)) as db:
        db.drop_tables()

        if extension == ".hdf5" or extension == ".h5":
            table = db.table("hdf5_raw")
            table.insert_multiple(samples)
        else:
            db.insert_multiple(samples)

    print(
        "--- Adding %d file(s) metadata into %s db took %s seconds ---\n"
        % (n_files, db_filename, (time.time() - start_time))
    )


def main():
    description = "Create db file from existing files' metadata."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--extension", type=str, default=".xrm", help="Input file extension"
    )

    parser.add_argument(
        "--param",
        type=str,
        action="append",
        help="Add experiment parameter as key=value",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="Tag pattern for filename interpretation\n"
        "    e.g: angle_energy_repetition "
        "to parse values from filename",
    )

    args = parser.parse_args()
    extension = args.extension
    pattern = args.pattern

    extra_param = []
    if args.param is not None:
        extra_param = args.param

    create_db(
        INDEX_FILENAME,
        pattern=pattern,
        extension=extension,
        extra_param=extra_param,
    )


if __name__ == "__main__":
    main()
