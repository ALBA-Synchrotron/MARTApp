# -*- coding: utf-8 -*-
#
# Copyright (C) 2014-2023 ALBA Synchrotron
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

from sdm.mistral.parser import create_db as create_db_from_txm
from sdm.mistral.parser import get_db_path
from sdm.mistral.scripts.create_db import create_db as create_db_from_xrm


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def remove_argument(parser, arg):
    """
    Method to remove a given argument (string name) from a parser.
    """
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


base_parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter, add_help=False
)

parse_txm = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_txm.add_argument(
    "--txm",
    type=str,
    help="TXM is the filename of the .txt script containing the commands used\n"
    "to perform the image acquisition by the BL09 TXM microscope",
)

parse_db = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_db.add_argument(
    "--db",
    type=str,
    default="index.json",
    help="Filename of the experiment Database",
)

parse_pattern = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_pattern.add_argument(
    "--pattern",
    type=str,
    help="Tag pattern for filename interpretation\n"
    "    e.g: angle_energy_repetition "
    "to parse values from filename",
)

parse_get_db = argparse.ArgumentParser(
    parents=[parse_txm, parse_db, parse_pattern], add_help=False
)

parse_table = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_table.add_argument(
    "--table_for_stack",
    type=str,
    default="hdf5_averages",
    help="DB table of image files to create the stacks"
    "(default: hdf5_averages)",
)


parse_update = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_update.add_argument(
    "-u",
    "--update_db",
    type=str2bool,
    default="True",
    help="Update DB with hdf5 records\n" "(default: True)",
)


parse_subfolders = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_subfolders.add_argument(
    "-s",
    "--subfolders",
    action="store_true",
    help="- If present: Use subfolders for indexing\n"
    "- Otherwise: Use general folder for indexing\n",
)

parse_cores = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_cores.add_argument(
    "-c",
    "--cores",
    type=int,
    default=-1,
    help="Number of cores used for the format conversion\n"
    "(default is max of available CPUs: -1)",
)

parse_crop = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_crop.add_argument(
    "--crop",
    type=str2bool,
    default="True",
    help="- If True: Crop images\n"
    "- If False: Do not crop images\n"
    "(default: True)",
)

parse_stack = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_stack.add_argument(
    "--stack",
    action="store_true",
    help="- If present: Calculate stack\n"
    "  Otherwise: Do not calculate stack",
)

parse_mrc = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_mrc.add_argument(
    "-m",
    "--hdf_to_mrc",
    type=str2bool,
    default="True",
    help="Convert FS hdf5 to mrc\n" "(default: True)",
)

parse_deconv = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_deconv.add_argument(
    "-d",
    "--deconvolution",
    action="store_true",
    help="- If present: Deconvolve mrc normalized stacks",
)

parse_ln = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_ln.add_argument(
    "-ln",
    "--minus_ln",
    type=str2bool,
    help="Compute absorbance stack\n" "(default: True)",
)

parse_align = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_align.add_argument(
    "-a",
    "--align",
    action="store_true",
    help="- If present: Align the different tomography projections",
)

parse_date = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_date.add_argument(
    "-d",
    "--date",
    type=int,
    default=None,
    help="Date of files to be normalized\n"
    "If None, no filter is applied\n"
    "(default: None)",
)

parse_sample = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_sample.add_argument(
    "-s",
    "--sample",
    type=str,
    default=None,
    help="Sample name of files to be normalized\n"
    "If None, all sample names are normalized\n"
    "(default: None)",
)

parse_energy = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_energy.add_argument(
    "-e",
    "--energy",
    type=float,
    default=None,
    help="Energy of files to be normalized\n"
    "If None, no filter is applied\n"
    "(default: None)",
)

parse_table_proc = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_table_proc.add_argument(
    "--table_h5",
    type=str,
    default="hdf5_proc",
    help="DB table of hdf5 to be processed\n"
    "If None, default tinyDB table is used\n"
    "(default: hdf5_proc)",
)

parse_title = argparse.ArgumentParser(parents=[base_parser], add_help=False)
parse_title.add_argument(
    "--title",
    type=str,
    default="X-ray tomography",
    help="Sets the title of the tomography",
)

parse_source_name = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_source_name.add_argument(
    "--source-name", type=str, default="ALBA", help="Sets the source name"
)

parse_source_type = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_source_type.add_argument(
    "--source-type",
    type=str,
    default="Synchrotron X-ray Source",
    help="Sets the source type",
)

parse_source_probe = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_source_probe.add_argument(
    "--source-probe",
    type=str,
    default="x-ray",
    help="Sets the source probe. Possible options "
    "are: 'x-ray', 'neutron', 'electron'",
)

parse_instrument_name = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_instrument_name.add_argument(
    "--instrument-name",
    type=str,
    default="BL09 @ ALBA",
    help="Sets the instrument name",
)

parse_sample_name = argparse.ArgumentParser(
    parents=[base_parser], add_help=False
)
parse_sample_name.add_argument(
    "--sample-name",
    type=str,
    default="Unknown",
    help="Sets the sample name",
)


def get_db(args):
    """
    Get or create db filename from args
    """
    if args.txm:
        db_filename = get_db_path(args.txm)
        create_db_from_txm(args.txm)
    else:
        db_filename = args.db
        if not os.path.exists(db_filename):
            if (
                len([file for file in os.listdir() if file.endswith(".xrm")])
                != 0
            ):
                create_db_from_xrm(
                    db_filename, pattern=args.pattern, extension=".xrm"
                )
            elif (
                len(
                    [
                        file
                        for file in os.listdir()
                        if file.endswith((".hdf5", ".h5"))
                    ]
                )
                != 0
            ):
                create_db_from_xrm(
                    db_filename, pattern=args.pattern, extension=".hdf5"
                )
            else:
                raise Exception(
                    "No TXM or HDF5 files found in the current directory."
                )

    return db_filename
 
