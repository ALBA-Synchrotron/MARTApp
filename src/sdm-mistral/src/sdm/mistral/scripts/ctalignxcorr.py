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
import time
from typing import Tuple

from sdm.mistral.scripts.extract_angle import extract_angles
from sdm.mistral.scripts.parser import base_parser, parse_mrc
from sdm.mistral.scripts.txrm2deconv import convert_hdf2mrc


WORKFLOW = "[hdf52mrc] >> align2D >> prealign >> stack >> align2D >> stack\n"


def app_parser():
    """Application parser."""
    description = (
        "This script is used to align stacks with fiducials." f"\n\n{WORKFLOW}"
    )

    parser = argparse.ArgumentParser(
        parents=[base_parser, parse_mrc], description=description
    )

    parser.add_argument(
        "hdf_norm_file",
        type=str,
        help="Enter hdf5 file containing the normalized tomography or "
        + "spectroscopy",
    )

    parser.add_argument(
        "mrc_file",
        type=str,
        nargs="?",
        default=None,
        help="Enter mrc file containing the deconvolved tomography or "
        + "spectroscopy",
    )

    parser.add_argument(
        "-a",
        "--angles",
        type=str,
        help="tlt file with the angles.",
        required=False,
    )

    parser.add_argument(
        "--n_fid",
        type=int,
        default=30,
        help="Approximated number of expected fiducials",
        required=False,
    )

    parser.add_argument(
        "--tilt_option",
        type=int,
        default=0,
        choices=[0, 2],
        help=(
            "TiltOption of tiltalign method. If 0, it will not compute the "
            "real angles; if 2, it will compute the real angles except 0."
        ),
        required=False,
    )

    return parser


def generate_tracka_files(
    template_folder: str, sample_name: str, rawtlt_file: str
):
    """Copy and prepare tracka files from templates."""
    os.system(f"cp {template_folder}tracka.* .")
    os.system(f"sed -i 's/<FILENAME>/{sample_name}/g' tracka.*")
    os.system(f"sed -i 's/<TILT_FILENAME>/{rawtlt_file}/g' tracka.*")


def find_initial_translational_alignment(
    mrc_file: str,
    tiltfile: str,
    output: str,
    rotation: float = 0.0,
    sigma1: float = 0.03,
    radius2: float = 0.25,
    sigma2: float = 0.05,
):
    """Find initial translational alignment between successive images.

    It uses tilt series and cross-correlation.
    """
    cmd = (
        "tiltxcorr "
        + f"-input {mrc_file}"
        + f" -output {output}"
        + f" -tiltfile {tiltfile}"
        + f" -rotation {rotation}"
        + f" -sigma1 {sigma1}"
        + f" -radius2 {radius2}"
        + f" -sigma2 {sigma2}"
    )
    print(cmd)
    os.system(cmd)


def prexf_2_prexg(
    prexf_file: str,
    nfit: int = 0,
):
    """Convert .prexf file to .prexg file using IMOD."""
    cmd = (
        "xftoxg "
        + f"-input {prexf_file}"
        + f" -nfit {nfit}"
        + f" -goutput {prexf_file[:-6]}.prexg"
    )
    print(cmd)
    os.system(cmd)


def preali_stack(
    sample_name: str, mrc_file: str, mode: int = 0, float_: int = 2
):
    """Generate prealigned stack."""
    cmd = (
        "newstack "
        + f"-input {mrc_file}"
        + f" -output {sample_name}.preali"
        + f" -mode {mode}"
        + f" -xform {sample_name}.prexg"
        + f" -float {float_}"
    )
    print(cmd)
    os.system(cmd)


def ali_stack(
    sample_name: str,
    mrc_file: str,
    offset: Tuple[int, int] = (0, 0),
    taper: Tuple[int, int] = (1, 0),
):
    """Align stack."""
    cmd = (
        "newstack "
        + f"-input {mrc_file}"
        + f" -output {sample_name}.ali"
        + f" -offset {offset[0]},{offset[1]}"
        + f" -xform {sample_name}.xf"
        + " -origin"
        + f" -taper {taper[0]},{taper[1]}"
    )
    print(cmd)
    os.system(cmd)


def make_seed_model(
    tracka: str, spacing: float = 0.85, peak: float = 1.0, number: int = 30
):
    """Make a seed model for tilt series fiducial tracking using IMOD."""
    cmd = (
        "autofidseed "
        + f"-track {tracka}"
        + f" -spacing {spacing}"
        + f" -peak {peak}"
        + f" -number {number}"
    )
    print(cmd)
    os.system(cmd)


def track_fiducials(tracka: str):
    """Track fiducials in a tilt series using IMOD."""
    cmd = "beadtrack " + f"-StandardInput < {tracka}"
    print(cmd)
    os.system(cmd)


def tilt_align(sample_name: str, rawtlt_file: str, tilt_option: int = 0):
    """Solve for alignment of tilted views using fiducials using IMOD."""
    print("TRACK FIDUCIALS: -------------------")
    cmd = (
        "tiltalign "
        + f" -ModelFile {sample_name}.fid"
        + f" -ImageFile {sample_name}.preali"
        + f" -OutputTiltFile {sample_name}.tlt"
        + f" -OutputTransformFile {sample_name}.tltxf"
        + " -RotationAngle 0.0"
        + f" -tiltfile {rawtlt_file}"
        + " -AngleOffset 0.0"
        + " -RotOption -1"
        + " -RotDefaultGrouping 5"
        + f" -TiltOption {tilt_option}"
        + " -MagReferenceView 1"
        + " -MagOption 0"
        + " -MagDefaultGrouping 4"
        + " -XStretchOption 0"
        + " -XStretchDefaultGrouping 7"
        + " -SkewOption 0"
        + " -SkewDefaultGrouping 11"
        + " -ResidualReportCriterion 3.0"
        + " -SurfacesToAnalyze 0"
        + " -MetroFactor 0.25"
        + " -MaximumCycles 1000"
        + " -KFactorScaling 0.7"
        + " -AxisZShift 0.0"
        + " -LocalAlignments 0"
        + " -MinFidsTotalAndEachSurface 8,3"
        + " -LocalOutputOptions 1,0,1"
        + " -LocalRotOption 3"
        + " -LocalRotDefaultGrouping 6"
        + " -LocalTiltOption 5"
        + " -LocalTiltDefaultGrouping 6"
        + " -LocalMagReferenceView 1"
        + " -LocalMagOption 3"
        + " -LocalMagDefaultGrouping 7"
        + " -LocalXStretchOption 0"
        + " -LocalXStretchDefaultGrouping 7"
        + " -LocalSkewOption 0"
        + " -LocalSkewDefaultGrouping 11"
        + " -BeamTiltOption 0"
    )
    print(cmd)
    os.system(cmd)


def transformations_product(prexg_file: str, tltxf_file: str, output: str):
    """Compute the product of two transformation files using IMOD."""
    cmd = (
        "xfproduct "
        + f"-in1 {prexg_file}"
        + f" -in2 {tltxf_file}"
        + f" -output {output}"
    )
    print(cmd)
    os.system(cmd)


def main(args=None):
    """Align images with fiducials."""
    parser = app_parser()
    args = parser.parse_args(args)

    print(WORKFLOW)

    start_time = time.time()

    # Load hdf_norm_file file
    hdf_norm_file = args.hdf_norm_file

    # Convert HDF5 to MRC if needed
    if args.mrc_file is None:
        mrc_file = convert_hdf2mrc(hdf_norm_file)
    else:
        mrc_file = args.mrc_file

    # Extract angle from HDF5 file and save it in angles.tlt
    if args.angles is None:
        print("\nEXTRACT ANGLES: -------------------")
        rawtlt_file = "angles.tlt"
        extract_angles(in_file_name=hdf_norm_file, out_file_name=rawtlt_file)
    else:
        rawtlt_file = args.angles

    # Delete "./" from input file if present
    sample_name = mrc_file.replace("./", "")
    sample_name = ".".join(sample_name.split(".")[:-1])

    # Copy templates and change the sample_name
    print("\nCOPY TEMPLATES: -------------------")
    # TODO: The path of the template folder is temporarily hardcoded,
    # TODO: but it shouldn't.
    template_folder = "/beamlines/bl09/controls/user-scripts/templates/"
    generate_tracka_files(
        template_folder=template_folder,
        sample_name=sample_name,
        rawtlt_file=rawtlt_file,
    )

    # Find initial translational alignment
    print("\nFIND INITIAL TRANSLATIONAL_ALIGNMENT: -------------------")
    find_initial_translational_alignment(
        mrc_file=mrc_file, tiltfile=rawtlt_file, output=sample_name + ".prexf"
    )

    # Transform .prexf transformation matrix to .prexg
    print("\nTRANSFORM PREXF TO PREXG: -------------------")
    prexf_2_prexg(prexf_file=sample_name + ".prexf")

    # Generate prealigned stack
    print("\nGENERATE PREALIGNED STACK: -------------------")
    preali_stack(sample_name=sample_name, mrc_file=mrc_file)

    # Generate seed model
    print("\nGENERATE SEED MODEL: -------------------")
    make_seed_model(tracka="tracka.com", number=args.n_fid)

    # Track fiducials
    print("\nTRACK FIDUCIALS: -------------------")
    track_fiducials(tracka="tracka.stdin")

    # Solve for alignment of tilted views using fiducial
    # Ignore error "TILTALIGN - No output file for local transforms specified"
    print("\nSOLVE FOR ALIGNMENT OF TILTED VIEWS: -------------------")
    tilt_align(
        sample_name=sample_name,
        rawtlt_file=rawtlt_file,
        tilt_option=args.tilt_option,
    )

    # Compute the product of two transformation files
    print("\nCOMPUTE THE PRODUCT OF TWO TRANSFORMATION: -------------------")
    transformations_product(
        prexg_file=sample_name + ".prexg",
        tltxf_file=sample_name + ".tltxf",
        output=sample_name + ".xf",
    )

    # Stack results
    print("\nSTACK RESULTS: -------------------")
    ali_stack(
        sample_name=sample_name, mrc_file=mrc_file, offset=(0, 0), taper=(1, 0)
    )

    print(f"\nExecution took {time.time() - start_time} seconds\n")


if __name__ == "__main__":
    main()
