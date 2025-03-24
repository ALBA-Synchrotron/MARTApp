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

import json
import os
from stat import S_ISREG, ST_CTIME, ST_MODE

import cv2
import h5py
import numpy as np


def sort_files_by_date(files):
    # get all entries in the directory w/ stats
    entries = ((os.stat(path), path) for path in files)
    # leave only regular files, insert creation date
    entries = (
        (stat[ST_CTIME], path)
        for stat, path in entries
        if S_ISREG(stat[ST_MODE])
    )
    # NOTE: on Windows `ST_CTIME` is a creation date
    #  but on Unix it could be something else
    # NOTE: use `ST_MTIME` to sort by a modification date
    sorted_files = []
    for cdate, path in sorted(entries):
        sorted_files.append(path)


def remove_all_from_db(db_filename, entry_name, list_values):
    with open(db_filename, "r") as f:
        data = json.load(f)

    to_delete = []
    for table in data:
        for entry in data[table]:
            if data[table][entry][entry_name] in list_values:
                to_delete.append((table, entry))

    for table, entry in to_delete:
        del data[table][entry]

    with open(db_filename, "w") as f:
        json.dump(data, f)


def select_roi(image_for_roi_selection, step=""):
    global roi_points
    roi_points = []

    mult_factor = 1 / image_for_roi_selection.max()
    image_for_roi_selection = mult_factor * image_for_roi_selection

    # Convert to float32 and to BGR to allow red ROIs
    image_for_roi_selection = image_for_roi_selection.astype("float32")
    image_for_roi_selection = cv2.cvtColor(
        image_for_roi_selection, cv2.COLOR_GRAY2BGR
    )

    def click_and_select(event, x, y, flags, param):
        # record the starting (x, y) coordinates (cols, rows)
        if event == cv2.EVENT_LBUTTONDOWN:
            global roi_points
            roi_points = [(x, y)]
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates (cols, rows)
            roi_points.append((x, y))
            # draw a rectangle around the region of interest
            cv2.rectangle(
                image_for_roi_selection,
                roi_points[0],
                roi_points[1],
                (0, 0, 255),
                2,
            )  # (0, 100, 0), 2)
            cv2.imshow(step, image_for_roi_selection)

    # load the image, clone it, and setup the mouse callback function
    clone = image_for_roi_selection.copy()
    cv2.namedWindow(step)
    cv2.setMouseCallback(step, click_and_select)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(step, image_for_roi_selection)
        key = cv2.waitKey(1) & 0xFF

        # Esc key to stop
        if key == 27:
            import sys

            sys.exit()
        # if the 'r' key is pressed, reset the selected region
        elif key == ord("r"):
            image_for_roi_selection = clone.copy()
            cv2.imshow(step, image_for_roi_selection)
        # if 'ENTER' key is pressed, store ROI coordinates and break loop
        elif key == 13 or key == ord("\u000A"):
            # close all open windows
            cv2.destroyWindow(step)
            break

    # Choose ROI points (invariant w.r.t. the selection direction)
    x0, y0 = roi_points[-2]
    x1, y1 = roi_points[-1]
    if x0 < x1 and y0 < y1:
        x_ini, x_fin, y_ini, y_fin = x0, x1, y0, y1
    elif x0 < x1 and y0 > y1:
        x_ini, x_fin, y_ini, y_fin = x0, x1, y1, y0
    elif x0 > x1 and y0 < y1:
        x_ini, x_fin, y_ini, y_fin = x1, x0, y0, y1
    elif x0 > x1 and y0 > y1:
        x_ini, x_fin, y_ini, y_fin = x1, x0, y1, y0
    else:
        pass
    roi_points = [(y_ini, x_ini), (y_fin, x_fin)]  # Flip coordinates.

    return roi_points


def save_ff_bg_stacks(destination_directory, ff_bg_stacks):
    stacks = [
        os.path.join(destination_directory, f)
        for f in os.listdir(destination_directory)
        if os.path.isfile(os.path.join(destination_directory, f))
        and any(
            substr in f
            for substr in [
                "norm_stack",
                "FFnorm_stack",
                "BGsub_stack",
                "FFnormBGsub_stack",
            ]
        )
    ]

    for key in ff_bg_stacks:
        destination_file = [str(f) for f in stacks if f"_{key}_" in f][0]
        with h5py.File(destination_file, "a") as stack:
            if "ff" in ff_bg_stacks[key]:
                # Preven problems with previous executions
                # del stack["TomoNormalized/FF_angles"]
                # del stack["TomoNormalized/FF"]

                stack.create_dataset(
                    "TomoNormalized/FF_angles",
                    data=np.array(list(ff_bg_stacks[key]["ff"].keys())),
                )
                stack.create_dataset(
                    "TomoNormalized/FF",
                    data=np.array(list(ff_bg_stacks[key]["ff"].values())),
                )

            if "bg" in ff_bg_stacks[key]:
                stack.create_dataset(
                    "TomoNormalized/BG_angles",
                    data=np.array(list(ff_bg_stacks[key]["bg"].keys())),
                )
                stack.create_dataset(
                    "TomoNormalized/BG",
                    data=np.array(list(ff_bg_stacks[key]["bg"].values())),
                )
