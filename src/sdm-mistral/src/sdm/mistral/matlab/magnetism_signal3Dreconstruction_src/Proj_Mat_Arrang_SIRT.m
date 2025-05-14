%Function to Rearrange the output of the parallelized loop in order to
%create the sparse projection matrix for a certain problem.
%--------------------------------------------------------------------------
%Function Inputs
%
%   Det_Size:       Row Vector containing the dimensions of the detector in
%                   X and Y directions.Ex:[200,200] 200 pixels in X and 200
%                   pixels in Y.
%   SimultSlcs:     Integer indicating the number of slices taken together
%                   to speedup the computation of the reconstruction by
%                   reducing the function execution calls.
%   Mod_Size:       Row Vector containing the dimensions of the
%                   reconstruction model along X, Y and Z directions.
%                   Number of model cells in X, Y and Z. Ex: [200,200,8]
%                   200 cells in X, 200 cells in Y and 8 cells in Z.
%   Recon_Flag:     Character array with three possible values indicating
%                   if the projection matrix to be calculated is associated
%                   to the Scalar or Vector reconstruction problem. For
%                   Scalar the flags are 'ScalarXtilt' and 'ScalarYtilt',
%                   and for the Vector, the flags are 'XTiltVector',
%                   'YTiltVector'.
%   Pix_Size:       Pixel Size in m in order to get the results in IS
%                   units.
%   Tilt_Ang:       Vector containing all the angles arranged as the slices
%                   to be recnstructed.This must be in degrees. It is
%                   considred counterclockwise sense around the rotation
%                   axis for positive vaules of the tilt agnle.
%   Det_idx_Cell_1: Cell array containing in each cell a number indicating
%                   the detector pixel taken into account for the ray
%                   interaction with the model. Information Sufficient for
%                   Scalar Reconstruction. In the dual tilt case it must be
%                   Ytilt data related.
%   Mod_idx_Cell_1: Cell array containing in each cell a matrix containing
%                   the model indices which interact with a certain ray.
%                   These indices are [row,column,layer] of the model.
%                   Information sufficient for Scalar Recontruction. In the
%                   dual tilt case it must be Ytilt data related.
%   Length_Cell_1:  Cell array cintaining in each cell a column vector with
%                   the lengths of a certain ray along all the interacting
%                   model cells. The number of elements of this vector must
%                   be the same thant the number of rows in the matrix of
%                   indices of Model_idx_Cell for the same cell structure
%                   index. Information Sufficient for Scalar
%                   Reconstruction. In the dual tilt case it must be Ytilt
%                   data related.
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
%Code created by Aurelio Hierro Rodriguez at ALBA Synchrotron (Spain).
%e-mail:    ahierro@cells.es/aurehr2001@gmail.com
%09/11/2017
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
function [Fwd_Proj_Mat,R_Mat,C_Mat] = Proj_Mat_Arrang_SIRT(Det_Data,SimultSlcs,Mod_Size,Recon_Flag,Pix_Size,Det_idx_Cell_1,Model_idx_Cell_1,Length_Cell_1,Ang_Cell_1)
%Flag Check
if strcmp(Recon_Flag,'ScalarXtilt') == 0 && strcmp(Recon_Flag,'ScalarYtilt') == 0 && strcmp(Recon_Flag,'XTiltVector') == 0 && strcmp(Recon_Flag,'YTiltVector') == 0 && strcmp(Recon_Flag,'FullVector') == 0
    fprintf('Error.\nThe Recon_Flag input argument must be ScalarXtilt, ScalarYtilt, XTiltVector, YTiltVector or FullVector only.')
    Fwd_Proj_Mat = nan(1);
    R_Mat = nan(1);
    C_Mat = nan(1);
    return
elseif strcmp(Recon_Flag,'ScalarXtilt') == 1 || strcmp(Recon_Flag,'ScalarYtilt') == 1
    Dat_n_pix = Det_Data(1)*Det_Data(3)*SimultSlcs;
    Mod_n_vox = Mod_Size(1)*Mod_Size(2)*Mod_Size(3);
    len_v = vertcat(Length_Cell_1{:})*Pix_Size;
    Mat_Row_idx = vertcat(Det_idx_Cell_1{:});
    Mat_Col_idx = cellfun(@(x) x(:,2)+(x(:,1)-1)*Mod_Size(1)+(x(:,3)-1)*Mod_Size(1)*Mod_Size(2),Model_idx_Cell_1,'UniformOutput',false);
    Mat_Col_idx = vertcat(Mat_Col_idx{:});
    %Control of the Matrix Dimensions
    if min(Mat_Row_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
    else
    end
    if min(Mat_Col_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
    else
    end
    if max(Mat_Row_idx) < Dat_n_pix
        Mat_Row_idx = [Mat_Row_idx;Dat_n_pix];
        Mat_Col_idx = [Mat_Col_idx;1];
        len_v = [len_v;0];
    else
    end
    if max(Mat_Col_idx) < Mod_n_vox
        Mat_Row_idx = [Mat_Row_idx;1];
        Mat_Col_idx = [Mat_Col_idx;Mod_n_vox];
        len_v = [len_v;0];
    else
    end

    Fwd_Proj_Mat = sparse(abs(Mat_Row_idx),abs(Mat_Col_idx),len_v,Dat_n_pix,Mod_n_vox);
    C_Mat = sparse(1:Mod_n_vox,1:Mod_n_vox, 1./sum(Fwd_Proj_Mat,1),Mod_n_vox,Mod_n_vox);
    R_Mat = sparse(1:Dat_n_pix,1:Dat_n_pix, 1./sum(Fwd_Proj_Mat,2),Dat_n_pix,Dat_n_pix);

elseif strcmp(Recon_Flag,'XTiltVector') == 1
    Dat_n_pix = Det_Data(1)*Det_Data(3)*SimultSlcs;
    Mod_n_vox = Mod_Size(1)*Mod_Size(2)*Mod_Size(3);
    len_v_Xt = vertcat(Length_Cell_1{:})*Pix_Size;
    Ang = vertcat(Ang_Cell_1{:})*(pi/180);
    Mat_Row_idx_my = vertcat(Det_idx_Cell_1{:});
    Mat_Row_idx_mz_Xt = Mat_Row_idx_my;
    Mat_Col_idx_my = cellfun(@(x) x(:,2)+(x(:,1)-1)*Mod_Size(1)+(x(:,3)-1)*Mod_Size(1)*Mod_Size(2),Model_idx_Cell_1,'UniformOutput',false);
    Mat_Col_idx_my = vertcat(Mat_Col_idx_my{:});
    Mat_Col_idx_mz_Xt = cellfun(@(x) Mod_n_vox+x(:,2)+(x(:,1)-1)*Mod_Size(1)+(x(:,3)-1)*Mod_Size(1)*Mod_Size(2),Model_idx_Cell_1,'UniformOutput',false);
    Mat_Col_idx_mz_Xt = vertcat(Mat_Col_idx_mz_Xt{:});
    %Data Rearrangement
    Mat_Row_idx = [Mat_Row_idx_my;Mat_Row_idx_mz_Xt];
    Mat_Col_idx = [Mat_Col_idx_my;Mat_Col_idx_mz_Xt];
    len_v_Norm = [len_v_Xt;len_v_Xt];
    len_v = [len_v_Xt.*sin(Ang);len_v_Xt.*(-cos(Ang))];
    %Control of the Matrix Dimensions
    if min(Mat_Row_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
        len_v_Norm = [0;len_v_Norm];
    else
    end
    if min(Mat_Col_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
        len_v_Norm = [0;len_v_Norm];
    else
    end
    if max(Mat_Row_idx) < Dat_n_pix
        Mat_Row_idx = [Mat_Row_idx;Dat_n_pix];
        Mat_Col_idx = [Mat_Col_idx;1];
        len_v = [len_v;0];
        len_v_Norm = [len_v_Norm;0];
    else
    end
    if max(Mat_Col_idx) < 2*Mod_n_vox
        Mat_Row_idx = [Mat_Row_idx;1];
        Mat_Col_idx = [Mat_Col_idx;2*Mod_n_vox];
        len_v = [len_v;0];
        len_v_Norm = [len_v_Norm;0];
    else
    end
    Fwd_Proj_Mat = sparse(abs(Mat_Row_idx),abs(Mat_Col_idx),len_v,Dat_n_pix,2*Mod_n_vox);
    Fwd_Proj_Mat_Norm = sparse(Mat_Row_idx,Mat_Col_idx,len_v_Norm,Dat_n_pix,2*Mod_n_vox);
    C_Mat = sparse(1:2*Mod_n_vox,1:2*Mod_n_vox, 1./sum(Fwd_Proj_Mat_Norm,1),2*Mod_n_vox,2*Mod_n_vox);
    R_Mat = sparse(1:Dat_n_pix,1:Dat_n_pix, 1./sum(Fwd_Proj_Mat_Norm,2),Dat_n_pix,Dat_n_pix);

elseif strcmp(Recon_Flag,'YTiltVector') == 1
    Dat_n_pix = Det_Data(1)*Det_Data(3)*SimultSlcs;
    Mod_n_vox = Mod_Size(1)*Mod_Size(2)*Mod_Size(3);
    len_v_Yt = vertcat(Length_Cell_1{:})*Pix_Size;
    Ang = vertcat(Ang_Cell_1{:})*(pi/180);
    Mat_Row_idx_mx = vertcat(Det_idx_Cell_1{:});
    Mat_Row_idx_mz_Yt = Mat_Row_idx_mx;
    Mat_Col_idx_mx = cellfun(@(x) x(:,2)+(x(:,1)-1)*Mod_Size(1)+(x(:,3)-1)*Mod_Size(1)*Mod_Size(2),Model_idx_Cell_1,'UniformOutput',false);
    Mat_Col_idx_mx = vertcat(Mat_Col_idx_mx{:});
    Mat_Col_idx_mz_Yt = cellfun(@(x) Mod_n_vox+x(:,2)+(x(:,1)-1)*Mod_Size(1)+(x(:,3)-1)*Mod_Size(1)*Mod_Size(2),Model_idx_Cell_1,'UniformOutput',false);
    Mat_Col_idx_mz_Yt = vertcat(Mat_Col_idx_mz_Yt{:});
    %Data Rearrangement
    Mat_Row_idx = [Mat_Row_idx_mx;Mat_Row_idx_mz_Yt];
    Mat_Col_idx = [Mat_Col_idx_mx;Mat_Col_idx_mz_Yt];
    len_v_Norm = [len_v_Yt;len_v_Yt];
    len_v = [len_v_Yt.*sin(Ang);len_v_Yt.*(-cos(Ang))];
    %Control of the Matrix Dimensions
    if min(Mat_Row_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
        len_v_Norm = [0;len_v_Norm];
    else
    end
    if min(Mat_Col_idx) > 1
        Mat_Row_idx = [1;Mat_Row_idx];
        Mat_Col_idx = [1;Mat_Col_idx];
        len_v = [0;len_v];
        len_v_Norm = [0;len_v_Norm];
    else
    end
    if max(Mat_Row_idx) < Dat_n_pix
        Mat_Row_idx = [Mat_Row_idx;Dat_n_pix];
        Mat_Col_idx = [Mat_Col_idx;1];
        len_v = [len_v;0];
        len_v_Norm = [len_v_Norm;0];
    else
    end
    if max(Mat_Col_idx) < 2*Mod_n_vox
        Mat_Row_idx = [Mat_Row_idx;1];
        Mat_Col_idx = [Mat_Col_idx;2*Mod_n_vox];
        len_v = [len_v;0];
        len_v_Norm = [len_v_Norm;0];
    else
    end
    Fwd_Proj_Mat = sparse(abs(Mat_Row_idx),abs(Mat_Col_idx),len_v,Dat_n_pix,2*Mod_n_vox);
    Fwd_Proj_Mat_Norm = sparse(Mat_Row_idx,Mat_Col_idx,len_v_Norm,Dat_n_pix,2*Mod_n_vox);
    C_Mat = sparse(1:2*Mod_n_vox,1:2*Mod_n_vox, 1./sum(Fwd_Proj_Mat_Norm,1),2*Mod_n_vox,2*Mod_n_vox);
    R_Mat = sparse(1:Dat_n_pix,1:Dat_n_pix, 1./sum(Fwd_Proj_Mat_Norm,2),Dat_n_pix,Dat_n_pix);

end


end
