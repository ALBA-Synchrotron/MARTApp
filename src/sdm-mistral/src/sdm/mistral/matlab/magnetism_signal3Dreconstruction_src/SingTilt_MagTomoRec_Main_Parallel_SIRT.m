%Main Function to get the magnetic reconstruction from a soft x-ray
%magnetic transmission microscopy tomogram. Function designed to be
%executed using the Matlab Runtime libraries allowing to use not Matlab
%licensed systems.
%--------------------------------------------------------------------------
%Function Inputs
%   Mag_Tomo:   Filename HDF5 containing the tomogram of the system to be
%               studied. It must contain the difference of positive and
%               negative dichroism datasets.
%   Tilt_Dataset:  Character array containing the inner hdf5 path with the 
%               information of the angles of projection associated to each 
%               image inside the HDF5 files with the tomograms. This 
%               implies that Ytilt and Xtilt must be measured under the
%               same incidence angles and in the same order. 
%               The data must be arranged into a column vector
%               inside the file (one angle each line) and it must be in 
%               degrees.
%   DataSet:    Character array containing the inner hdf5 path with the
%               tomographic data.(In MISTRAL the standard is '/NXtomo/data/data')
%   Mod_SX:     Integer indicating the number of model cells in X
%               direction. It can be as large as the number of columns in
%               the input tomogram.
%   Mod_SY:     Integer indicating the number of model cells in Y
%               direction. It can be as large as the number of rows in
%               the input tomogram.
%   Mod_SX:     Integer indicating the number of model cells in Z
%               direction. It can be as large as one consider because it is
%               related with the numbe rof layers to be reconstructed
%               in-depth.
%   Pix_Size:   Number indicating the pixel size in the detector in m.
%   N_iter:     Number of iterations to sotp the SIRT algorithm and get the
%               reconstruction.
%   LC_Flag:    Logical value(1 or 0) allowing to reconstruct for a
%               Continuous Film or an isolated structure. 1 means LC
%               enabled and 0 disabled.
%   Recon_Flag: Character array indicating if the vector reconstruction is 
%               performed by rotating around the X oYXthe Y axis. The first
%               means an stretching in the vertical disdfsrection and the
%               second one in the horizontal direction. The flag can take
%               'XTiltVector' or 'YTiltVector'.
%   SimultSlcs: Integer number indicating the number of simultaneous slides
%               to be reconstructed. This increases the performance of the
%               reconstruction if there is enough Memory (either CPU or
%               GPU) available to speed-up the reconstruction. Ex: 3
%               reconstructs grouping the data in blocks of 3 consecutive
%               Slices of the tomogram.
%   GPU_Flag:   Logical flag to use a GPU to improve the performance of the
%               reconstruction. 1 (enabled) 0 (disabled). It only works if
%               a Nvidia GPU is available with CUDA instaled.
%   SaveCalMat: Logical flag to enable (1) or disable (0) the saving of the 
%               projection matrices.
%   Use_CalMat: Logical flag to use or not pre-calculated reconstruction
%               matrices. These should be stored inside the working folder
%               in a folder called 'Projection_Matrices'. The name of each
%               block of matrices should be Projection_x.mat with x being
%               the projection number. 1 is enabled 0 is disbled.
%   Filename_Out:   String indicating the name of the output generated hdf5
%                   file with the reconstruction results.
%   Mask_Recon_File: Optional. Character array containing the hdf5 file
%               with the mask applied in the reconstruction.
%   Mask_Recon_Dataset: Optional. Character array containing the inner hdf5 
%               path with the mask applied in the reconstruction.
%
%Function Outputs
%
%   The function will generate ONE HDF5 file called
%   "Filename_Out_MagneticReconstruction.hdf5" containing the
%   reconstructions obtained for magnetization components 1 and 2.
%   The sieze of the generated files will be the size of the model
%   indicated in Mod_Size variable.
%
%--------------------------------------------------------------------------
%Code created by Aurelio Hierro Rodriguez at MCMP group at University of 
%Glasgow.
%e-mail:    Aurelio.HierroRodriguez@glasgow.ac.uk / aurehr2001@gmail.com
%26/06/2018
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
%--------------------------------------------------------------------------
function SingTilt_MagTomoRec_Main_Parallel_SIRT(Mag_Tomo,Tilt_Dataset,DataSet,Mod_SX,Mod_SY,Mod_SZ,Pix_Size,N_iter,LC_Flag,Recon_Flag,SimultSlcs,GPU_Flag,SaveCalMat,Use_CalMat,Filename_Out,varargin)
if strcmp(Recon_Flag,'XTilt') == 0 && strcmp(Recon_Flag,'YTilt') == 0
    fprintf('Error.\nThe Recon_Flag input argument must be XTilt or YTilt only.')
    return
end
%Initilaizing Parallel Configuration (not used if using the precompiled configuration)
% profile_master = parallel.importProfile('local');
% parallel.defaultClusterProfile(profile_master);
% defaultProfile = parallel.defaultClusterProfile;
% myCluster = parcluster(defaultProfile);
% parexec = parpool(myCluster);
%Tomogram Reading
M_Tomo = double(h5read(Mag_Tomo,strcat('/',DataSet)));
%Angle data Reading
Tilt_Ang = double(h5read(Mag_Tomo,strcat('/',Tilt_Dataset)));
%String 2 number
Pix_Size = str2double(Pix_Size);
N_iter = str2double(N_iter);
LC_Flag = str2double(LC_Flag);
Recon_Flag = strcat(Recon_Flag,'Vector');
SimultSlcs = str2double(SimultSlcs);
GPU_Flag = str2double(GPU_Flag);
SaveCalMat = str2double(SaveCalMat);
Use_CalMat = str2double(Use_CalMat);
Mod_SX = str2double(Mod_SX);
Mod_SY = str2double(Mod_SY);
Mod_SZ = str2double(Mod_SZ);
Mod_Size = [Mod_SX,Mod_SY,Mod_SZ];
if nargin < 16
    Recon_Mask = ones(size(M_Tomo,1),size(M_Tomo,2),Mod_SZ);
elseif nargin < 17
    Recon_Mask = double(h5read(Mag_Tomo,strcat('/',varargin{1})));
else
    Recon_Mask = double(h5read(varargin{1},strcat('/',varargin{2})));
end
pad = (size(M_Tomo(:,:,1))-[Mod_SY,Mod_SX])/2;
xrange = max(floor(pad(2))+1,1):min(floor(pad(2))+Mod_SX,size(M_Tomo,2));
yrange = max(floor(pad(1))+1,1):min(floor(pad(1))+Mod_SY,size(M_Tomo,1));
M_Tomo = M_Tomo(yrange,xrange,:);
pad = (size(Recon_Mask(:,:,1))-[Mod_SY,Mod_SX])/2;
xrange = max(floor(pad(2))+1,1):min(floor(pad(2))+Mod_SX,size(Recon_Mask,2));
yrange = max(floor(pad(1))+1,1):min(floor(pad(1))+Mod_SY,size(Recon_Mask,1));
Recon_Mask = Recon_Mask(yrange,xrange,:);

%Launching the Recontruction Algorithm
 [m1_Mod,m2_Mod] = VecTomoRec_Parallel_SingTilt_SIRT(Recon_Mask,M_Tomo,Pix_Size,Tilt_Ang,Mod_Size,N_iter,LC_Flag,Recon_Flag,SimultSlcs,GPU_Flag,Use_CalMat,SaveCalMat);
if exist(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out),'file')
    delete(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out))
end
%Creating the Output Files
h5create(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out),'/MagneticReconstruction/M1',[Mod_Size(2),Mod_Size(1),Mod_Size(3)]);
h5create(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out),'/MagneticReconstruction/M2',[Mod_Size(2),Mod_Size(1),Mod_Size(3)]);
%Writing the Output Files
h5write(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out),'/MagneticReconstruction/M1',m1_Mod);
h5write(sprintf('%s_MagneticReconstruction.hdf5',Filename_Out),'/MagneticReconstruction/M2',m2_Mod);
%Closing the Pool
% delete(parexec)
return
end