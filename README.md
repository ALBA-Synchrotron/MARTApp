# MARTApp

**MARTApp (Magnetic Analysis & Reconstruction of Tomographies App)** is a cross-platform application developed in BL09 MISTRAL for the analysis and vectorial reconstruction of magnetic tomographies acquired at the beamline (but not limited to it). 

In this repo, you may find:

- [**User guide** to use MARTApp in/outside ALBA](./user_guide.pdf)
- [**Quick Launch Manual** to launch the app outside ALBA](./quick_launch_manual.md)
- [**Source code**](./src)

## Data

<details>
<summary>Preprocessing input</summary>
To start from the pre-processing stage, the data must consist of one Xradia (Zeiss microCT) XRM file per acquisition (polarization-angle-repetition) or one
HDF5 file (`*.h5` or `*.hdf5`) with the following structure:

```
/                               Group
    /data                       Soft Link {data_1}
    /data_1                     Dataset {512, 512}
    /metadata                   Group
        /FF                     Dataset {SCALAR}
        /angle                  Dataset {SCALAR}
        /data_type              Dataset {SCALAR}
        /date_time_acquisition  Dataset {SCALAR}
        /energy                 Dataset {SCALAR}
        /exposure_time          Dataset {SCALAR}
        /image_height           Dataset {SCALAR}
        /image_width            Dataset {SCALAR}
        /instrument             Dataset {SCALAR}
        /machine_current        Dataset {SCALAR}
        /magnification          Dataset {SCALAR}
        /output_file            Dataset {SCALAR}
        /pixel_size             Dataset {SCALAR}
        /polarisation           Dataset {SCALAR}
        /sample_name            Dataset {SCALAR}
        /source                 Dataset {SCALAR}
        /source_probe           Dataset {SCALAR}
        /source_type            Dataset {SCALAR}
        /x_position             Dataset {SCALAR}
        /y_position             Dataset {SCALAR}
        /z_position             Dataset {SCALAR}
```
</details>

<details>
<summary>XMCD input (preprocessing output)</summary>
Preprocessing generates one HDF5 file per polarization.
Each file is structured as follows:

```
/                            Group
    /TomoNormalized          Group
        /Currents            Dataset {N}
        /ExpTimes            Dataset {N}
        /TomoNormalized      Dataset {N, H, W}
        /energy              Dataset {N}
        /polarisation        Dataset {1}
        /rotation_angle      Dataset {N}
        /x_pixel_size        Dataset {1}
        /y_pixel_size        Dataset {1}
```
</details>

<details>
<summary>Reconstruction input (XMCD output)</summary>
A single HDF5 file is produced by the XMCD stage and used for the different 
reconstructions:

```
/                               Group
    /2DAlignedNegativeStack     Dataset {N, H, W}
    /2DAlignedPositiveStack     Dataset {N, H, W}
    /Absorption2DAligned        Dataset {N, H, W}
    /Angles                     Dataset {N}
    /MagneticSignal2DAligned    Dataset {N, H, W}
    /OriginalNegativeStack      Dataset {N, H, W}
    /OriginalPositiveStack      Dataset {N, H, W}

```
</details>

<details>
<summary>Reconstruction output</summary>
Each reconstruction produces a single final HDF5 file.

Magnetic reconstruction of 2D samples:
```
/                               Group
    /mx                         Dataset {X, Y}
    /my                         Dataset {X, Y}
    /mz                         Dataset {X, Y}
    /r2m                        Dataset {X, Y}
```

Absorption reconstruction of 3D samples (two files, one per tilt series):
```
/                               Group
    /Absorption3D               Dataset {Z, X, Y}
    /Mask3D                     Dataset {Z, X, Y}
    /Mask3DRegistration         Dataset {Z, X, Y}
```

Magnetic reconstruction of 3D samples:
```
/                               Group
    /mx                         Dataset {Z, X, Y}
    /my                         Dataset {Z, X, Y}
    /mz                         Dataset {Z, X, Y}
```
</details>

## Do you have suggestions or have you found a bug?
If you have a suggestion for a feature that you would like to see included in the
application or something that could be improved, as well as in the case that you
found a bug, do not hesitate to open an issue using 
the Issues tab or to write an email to *jgsanchez (at) cells.es*.

## License
*MARTApp* is licensed under GPL v3.

## Cite & Reference
Please, if you use this application for the analysis/reconstruction of your data, remember to cite the paper that accompanies the software:

```
Herguedas-Alonso, A.E., Gómez Sánchez, J., Fernández-González, C., Sorrentino, A.,
Ferrer, S., Pereiro, E. & Hierro-Rodriguez, A. (2025). J. Synchrotron Rad.
32, https://doi.org/10.1107/S1600577525004485.
```

and the paper of the algorithm used for the reconstruction:

```
Herguedas-Alonso, A. E., Aballe, L., Fullerton, J., Vélez, M.,
Martín, J. I., Sorrentino, A., ... & Hierro-Rodriguez, A. (2023).
A fast magnetic vector characterization method for quasi two-dimensional systems and heterostructures.
Scientific Reports, 13(1), 9639.
```

```
Hierro-Rodriguez, A., Gürsoy, D., Phatak, C., Quirós, C.,
Sorrentino, A., Álvarez-Prado, L. M., ... & Ferrer, S. (2018).
3D reconstruction of magnetization from dichroic soft X-ray transmission tomography.
Journal of synchrotron radiation, 25(4), 1144-1152.
```

