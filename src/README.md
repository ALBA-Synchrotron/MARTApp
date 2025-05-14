# About MARTApp's code

In order to make this software open source and accessible for inspection, improvement, or modification, 
the application source code is available here. However, if you are not a developer, we recommend using the Docker image.

MARTApp is based on two python packages: `sdm-mistral`, which is the core and 
has the different used pipelines; and `martapp`, which only manages the application's front-end.

To use this code, please, create the required conda environment with the `environment.yml` file and install [artis-tomo](https://github.com/ALBA-Synchrotron-Methodology/artis_tomo), [sdm-mistral](./sdm-mistral/), and [martapp](./martapp/) packages. Then, simply call `martapp` in the command line.

To have full access to the application you will need to have [IMOD](https://bio3d.colorado.edu/imod/) (for some alignment algorithms) and [MATLAB Runtime 2017](https://es.mathworks.com/products/compiler/matlab-runtime.html) (for some reconstruction algorithms) installed. We include MATLAB code compiled and non-compiled.

Please, do not hesitate to contact us if you need help with the installation or if you find a bug.