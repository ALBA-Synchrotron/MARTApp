%Function to calculate the Forward/Backward projection matrix for a
%specific tilt angle in order to apply the SIRT algorithm. Scalar Case.
%CPU Parallelized.
%--------------------------------------------------------------------------
%Function Inputs
%
%   Det_Size:   Row Vector containing the dimensions of the detector in X
%               and Y directions.Ex:[200,200] 200 pixels in X and 200
%               pixels in Y.  
%   Det_Data:   Row Vector containing the information about the Fast
%               samplig direction size and the slice counter for the SIRT
%               reconstruction. Ex: [512,2] corresponds to a detector line
%               with 512 pixels and the matrices are going to be calculated
%               at the 2nd slice position.
%   Mod_Size:   Row Vector containing the dimensions of the reconstruction
%               model along X, Y and Z directions. Number of model cells in
%               X, Y and Z. Ex: [200,200,8] 200 cells in X, 200 cells in Y
%               and 8 cells in Z.
%   Pix_Size:   Pixel Size in m in order to get the results in IS units.
%   Tilt_Ang:   Column vector containing all the angles associated with the
%               tomogram. These must be entered in degrees.
%   Recon_Flag: Character array with three possible values indicating if
%               the projection matrix to be calculated is associated to the
%               reconstruction problem. The flags are 'XTilt' or 
%               'YTilt'.
%   SimultSlcs: Integer number indicating the number of simultaneous slides
%               to be reconstructed. This increases the performance of the
%               reconstruction if there is enough Memory (either CPU or
%               GPU) available to speed-up the reconstruction. Ex: 3
%               reconstructs grouping the data in blocks of 3 consecutive
%               Slices of the tomogram.
%   LC_Flag:    Logical input (1 or 0) indicating if we are dealing with a
%               reconstruction of a Continuous film with Missing Wedge
%               (angle of incidence below 75 deg, in order to avoid
%               extremly large models). 1 means activated, 0 means
%               deactivated.
%   Varargin(1):    Maximum incidence angle of the Missing Wedge Tomogram
%                   measured in degrees and in absolute value. It only has
%                   effect if the LC_Flag is active.
%
%Function Outputs
%
%   Fwd_Proj_Mat:   Sparse Matrix with the weight of each cell contributing
%                   for each pixel in the detector. The matrix has
%                   dimensions Number_Pixels_Det x Number_Model_Cells for
%                   the Scalar case. The matrix is compatible with the
%                   problem written as: "y - Ax = 0" where "y" represent a
%                   column vector containing all the Detector pixels
%                   arranged in raster order. y(i,j) -> moving through j
%                   first and after through i. "x" represents a column
%                   vector containing the information of all the model
%                   cells also arranged in raster order. x(m_i,m_j,m_k) ->
%                   moving through m_j, after through m_i and finally
%                   through m_k. To convert to real units it is necessary to
%                   multiply the result by the proper length of each pixel.
%   R_Mat:          Sparse Diagonal Matrix containing the sum of all the
%                   rows present in the Fwd_Proj_Mat. This matrix has
%                   renormalization purposes during SIRT algorithm
%                   execution.
%   C_Mat:          Sparse Diagonal Matrix containing the sum of all the
%                   columns present in the Fwd_Proj_Mat. This matrix has
%                   renormalization purposes during SIRT algorithm
%                   execution.
%
%--------------------------------------------------------------------------
%Code created by Aurelio Hierro Rodriguez at University of Oviedo (Spain).
%e-mail:    hierroaurelio@uniovi.es/aurehr2001@gmail.com
%2/8/2021
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
function [Fwd_Proj_Mat,R_Mat,C_Mat] = SclTomo_Fwd_Mat_Parallel_SingTiltSIRT(Det_Size,Det_Data,Pix_Size,Mod_Size,Tilt_Ang,Recon_Flag,SimultSlcs,LC_Flag,varargin)
%Flag Check
if strcmp(Recon_Flag,'XTilt') == 0 && strcmp(Recon_Flag,'YTilt') == 0
    fprintf('Error.\nThe Recon_Flag input argument must be XTilt, YTilt only.')
    Fwd_Proj_Mat = nan(1);
    R_Mat = nan(1);
    C_Mat = nan(1);
    return
else
    n_Proj = length(Tilt_Ang);
    %Detector position calculation. (Pixels)
    x_ds = linspace(-Det_Size(1)/2+0.5,Det_Size(1)/2-0.5,Det_Size(1));
    y_ds = linspace(-Det_Size(2)/2+0.5,Det_Size(2)/2-0.5,Det_Size(2));
    [x_d,y_d] = meshgrid(x_ds,y_ds);
    %Column Vector writing of the matrices x_d and y_d
    if strcmp(Recon_Flag,'XTilt') == 1
        x_d_v = zeros(Det_Size(2)*n_Proj*SimultSlcs,1);
        y_d_v = zeros(Det_Size(2)*n_Proj*SimultSlcs,1);
        for i=1:n_Proj
            x_d_v((i-1)*Det_Size(2)*SimultSlcs+1:(i-1)*Det_Size(2)*SimultSlcs+Det_Size(2)*SimultSlcs,1) = reshape(x_d(:,(Det_Data(2)-1)*SimultSlcs+1:Det_Data(2)*SimultSlcs),[Det_Size(2)*SimultSlcs,1]);
            y_d_v((i-1)*Det_Size(2)*SimultSlcs+1:(i-1)*Det_Size(2)*SimultSlcs+Det_Size(2)*SimultSlcs,1) = reshape(y_d(:,(Det_Data(2)-1)*SimultSlcs+1:Det_Data(2)*SimultSlcs),[Det_Size(2)*SimultSlcs,1]);
        end
    elseif strcmp(Recon_Flag,'YTilt') == 1
        x_d_v = zeros(Det_Size(1)*n_Proj*SimultSlcs,1);
        y_d_v = zeros(Det_Size(1)*n_Proj*SimultSlcs,1);
        for i=1:n_Proj
            x_d_v((i-1)*Det_Size(1)*SimultSlcs+1:(i-1)*Det_Size(1)*SimultSlcs+Det_Size(1)*SimultSlcs,1) = reshape(x_d((Det_Data(2)-1)*SimultSlcs+1:Det_Data(2)*SimultSlcs,:)',[Det_Size(1)*SimultSlcs,1]);
            y_d_v((i-1)*Det_Size(1)*SimultSlcs+1:(i-1)*Det_Size(1)*SimultSlcs+Det_Size(1)*SimultSlcs,1) = reshape(y_d((Det_Data(2)-1)*SimultSlcs+1:Det_Data(2)*SimultSlcs,:)',[Det_Size(1)*SimultSlcs,1]);
        end
    end
    if LC_Flag == 0
        Mod_SizeAr = Mod_Size;
    elseif LC_Flag == 1 
        max_Ang = varargin{(1)};
        ext_Mod = ceil(Mod_Size(3)*tand(max_Ang));
        if strcmp(Recon_Flag,'XTilt') == 1
            Mod_SizeAr = [Mod_Size(1),Mod_Size(2)+2*ext_Mod,Mod_Size(3)];
        elseif strcmp(Recon_Flag,'YTilt') == 1
            Mod_SizeAr = [Mod_Size(1)+2*ext_Mod,Mod_Size(2),Mod_Size(3)];
        else
        end
    else
        fprintf('Error. LC_Flag must be 1(activated) or 0(deactivated).\n')
        Fwd_Proj_Mat = [];
        R_Mat = [];
        C_Mat = [];
        return
    end
end
%XTilt
if strcmp(Recon_Flag,'XTilt') == 1
    Data_n_Pix = Det_Data(1)*n_Proj*SimultSlcs;
    Fast_Idx = Det_Data(1)*SimultSlcs;
    %z_vox = Mod_Size(3);
    Det_idx_Cell_Xt = cell(1,Data_n_Pix);
    Model_idx_Cell_Xt = cell(1,Data_n_Pix);
    Length_Cell_Xt = cell(1,Data_n_Pix);
    Ang_Cell_Xt = cell(1,Data_n_Pix);
    parfor i=1:Data_n_Pix
        ang_idx = ceil(i/Fast_Idx);
        d_p_v = [x_d_v(i),y_d_v(i)];
        [x_idx,y_idx,z_idx,len_v,~,~] = Raypath_Gen(d_p_v(1),d_p_v(2),...
            [1,0,0],[0,0,0],Tilt_Ang(ang_idx),Mod_SizeAr,[1,1,1],[0,0,1],...
            [0,0,0],0);
        if isempty(len_v) == 1
            Model_idx_Cell_Xt{i} = zeros(0,3);
        else
            Det_idx_Cell_Xt{i} = ones(length(len_v),1)*i;
            Model_idx_Cell_Xt{i} = [y_idx,x_idx,z_idx];
            Length_Cell_Xt{i} = len_v;
            Ang_Cell_Xt{i} = ones(size(len_v))*Tilt_Ang(ang_idx);
        end
    end
    [Fwd_Proj_Mat,R_Mat,C_Mat] = Proj_Mat_Arrang_SIRT([Det_Data,n_Proj],SimultSlcs,Mod_SizeAr,'ScalarXtilt',Pix_Size,Det_idx_Cell_Xt,Model_idx_Cell_Xt,Length_Cell_Xt,Ang_Cell_Xt);
%YTilt    
elseif strcmp(Recon_Flag,'YTilt') == 1
    Data_n_Pix = Det_Data(1)*n_Proj*SimultSlcs;
    Fast_Idx = Det_Data(1)*SimultSlcs;
    %z_vox = Mod_Size(3);
    Det_idx_Cell_Yt = cell(1,Data_n_Pix);
    Model_idx_Cell_Yt = cell(1,Data_n_Pix);
    Length_Cell_Yt = cell(1,Data_n_Pix);
    Ang_Cell_Yt = cell(1,Data_n_Pix);
    parfor i=1:Data_n_Pix
        ang_idx = ceil(i/Fast_Idx);
        d_p_v = [x_d_v(i),y_d_v(i)];
        [x_idx,y_idx,z_idx,len_v,~,~] = Raypath_Gen(d_p_v(1),d_p_v(2),...
            [0,1,0],[0,0,0],Tilt_Ang(ang_idx),Mod_SizeAr,[1,1,1],[0,0,1],...
            [0,0,0],0);
        if isempty(len_v) == 1
            Model_idx_Cell_Yt{i} = zeros(0,3);
        else
            Det_idx_Cell_Yt{i} = ones(length(len_v),1)*i;
            Model_idx_Cell_Yt{i} = [y_idx,x_idx,z_idx];
            Length_Cell_Yt{i} = len_v;
            Ang_Cell_Yt{i} = ones(size(len_v))*Tilt_Ang(ang_idx);
        end
    end
    [Fwd_Proj_Mat,R_Mat,C_Mat] = Proj_Mat_Arrang_SIRT([Det_Data,n_Proj],SimultSlcs,Mod_SizeAr,'ScalarYtilt',Pix_Size,Det_idx_Cell_Yt,Model_idx_Cell_Yt,Length_Cell_Yt,Ang_Cell_Yt);
else
end

end
