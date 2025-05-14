%Function for the Reconstruction of the Magnetisation compoents
%perpendicular to the rotation axis of the Tilt Series using the SIRT
%algorithm.
%--------------------------------------------------------------------------
%Function Inputs
%   Recon_Mask: Binary tensor with the dimensions indicated by Mod_Size
%               to constrain the magnetic reconstruction to the charge
%               volume of the sample.
%   M_Tomo:     Images Stack containing the Tilt series associated with the
%               difference of log(C+) and log(C-). These will contain the
%               XMCD contribution of the projections allowing to
%               reconstruct the magnetisation perpendicular to the rotation
%               axis. Each image corresponds to a different
%               projection angle and are in the order indicated by the
%               variable Tilt_v.
%   Pix_Size:   Number indicating the lateral size of a pixel in the
%               tomograms expressed in m.
%   Tilt_v:     Vector with dimension [N_project x 1] containing the
%               information of the tilt angle of each projection. It is
%               assumed that both Xtilt and Ytilt series have the same
%               projection angles.
%   Mod_Size:   Row Vector containing the dimensions of the reconstruction
%               model along X, Y and Z directions. Number of model cells in
%               X, Y and Z. Ex: [200,200,8] 200 cells in X, 200 cells in Y
%               and 8 cells in Z.
%   N_iter:     Integer indicating the number of iterations in order to
%               apply the SIRT algorithm. It should be between 20-80.
%   LC_Flag:    Logical input (1 or 0) indicating if we are dealing with a
%               reconstruction of a Continuous film with Missing Wedge
%               (angle of incidence below 75 deg, in order to avoid
%               extremly large models). 1 means activated, 0 means
%               deactivated.
%   Recon_Flag: Character array indicating if the vector reconstruction is
%               performed by rotating around the X or the Y axis. The first
%               means an stretching in the vertical direction and the
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
%               a Nvidia GPU is available with CUDA installed.
%   Use_CalMat: Logical flag to use or not pre-calculated reconstruction
%               matrices. These should be stored inside the working folder
%               in a folder called 'Projection_Matrices'. The name of each
%               block of matrices should be Projection_x.mat with x being
%               the projection number. 1 is enabled 0 is disbled.
%   SaveCalMat: Logical flag to enable (1) or disable (0) the saving of the
%               projection matrices.
%
%Function Outputs
%
%   m1_Mod:     Tensor with the dimensions indicated by Mod_Size containing
%               the reconstructed in-plane component of the magnetization
%               vector.
%   m2_Mod:     Tensor with the dimensions indicated by Mod_Size containing
%               the reconstructed out-of-plane component of the 
%               magnetization vector.
%
%--------------------------------------------------------------------------
%Code created by Aurelio Hierro Rodriguez at MCMP group of University of
%Glasgow.
%e-mail:    Aurelio.HierroRodriguez@glasgow.ac.uk/aurehr2001@gmail.com
%27/06/2018
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
function [m1_Mod,m2_Mod] = VecTomoRec_Parallel_SingTilt_SIRT(Recon_Mask,M_Tomo,Pix_Size,Tilt_v,Mod_Size,N_iter,LC_Flag,Recon_Flag,SimultSlcs,GPU_Flag,Use_CalMat,SaveCalMat)
if strcmp(Recon_Flag,'XTiltVector') == 0 && strcmp(Recon_Flag,'YTiltVector') == 0
    fprintf('Error.\nThe Recon_Flag input argument must be XTiltVector or YTiltVector only.')
    m1_Mod = nan(1);
    m2_Mod = nan(1);
    return
else
end
M_Tomo = -1.*M_Tomo;
n_Proj = length(Tilt_v);
Det_Size = size(M_Tomo);
Det_Size =[Det_Size(2),Det_Size(1),Det_Size(3)];
%Images Arrangement in Vector.
if strcmp(Recon_Flag,'XTiltVector') == 1 %Creation of the sinogram for each Y slice.
    %Checking the Number of Slices of the data matches the length of the
    %data.
    if mod(Det_Size(1),SimultSlcs) == 0
        %Do nothing everything is fine.
    else
        fprintf('Warning. Selected Number of Simultaneous Slices is not multiple of Number of Columns (%i).\n',Det_Size(1))
        fprintf('Defaulting to SIRT using 1 slice each time.\')
        SimultSlcs = 1;
    end
    M_Tomo_v = zeros(Det_Size(2)*SimultSlcs*n_Proj,Det_Size(1)/SimultSlcs);
    Fast_Idx = Det_Size(2); %Sinogram Line along Y.
    Slice_Count = Det_Size(1)/SimultSlcs; %Different slices along X direction.
    for ColBlock=1:Det_Size(1)/SimultSlcs
        for pj=1:n_Proj
            M_Tomo_v((pj-1)*Det_Size(2)*SimultSlcs+1:pj*Det_Size(2)*SimultSlcs,ColBlock) =reshape(M_Tomo(:,(ColBlock-1)*SimultSlcs+1:ColBlock*SimultSlcs,pj),[Det_Size(2)*SimultSlcs,1]);%./2; %The 2 factor is assuming that you are using log(Cp/Cm) as input data for the Magnetic Contribution.
        end
    end
elseif strcmp(Recon_Flag,'YTiltVector') == 1 %Creation of the sinogram for each X slice.
    %Checking the Number of Slices of the data matches the length of the
    %data.
    if mod(Det_Size(2),SimultSlcs) == 0
        %Do nothing everything is fine.
    else
        fprintf('Warning. Selected Number of Simultaneous Slices is not multiple of Number of Rows (%i).\n',Det_Size(2))
        fprintf('Defaulting to SIRT using 1 slice each time.\n')
        SimultSlcs = 1;
    end
    M_Tomo_v = zeros(Det_Size(1)*SimultSlcs*n_Proj,Det_Size(2)/SimultSlcs);
    Fast_Idx = Det_Size(1); %Sinogram Line along X.
    Slice_Count = Det_Size(2)/SimultSlcs; %Different slices along Y direction.
    for RowBlock=1:Det_Size(2)/SimultSlcs
        for pj=1:n_Proj
            M_Tomo_v((pj-1)*Det_Size(1)*SimultSlcs+1:pj*Det_Size(1)*SimultSlcs,RowBlock) =reshape(M_Tomo((RowBlock-1)*SimultSlcs+1:RowBlock*SimultSlcs,:,pj)',[Det_Size(1)*SimultSlcs,1]);%./2;
        end
    end
end
clear('M_Tomo');
if GPU_Flag == 0 %No GPU Implementation.
    if LC_Flag == 0
        ext_Mod = 0;
        %Model Initialization
        M_Mod_v = zeros(2*Mod_Size(1)*Mod_Size(2)*Mod_Size(3),1);
        if Use_CalMat == 0
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1 %Saving the projection matrices in the HDD to save execution time.
                        if slc == 1 && SaveCalMat == 1
                            mkdir('Mag_Projection_Matrices')
                        else
                        end
                        if SaveCalMat == 1 %Save the projection matrices.
                            path_mat = pwd;
                            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            save(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','R_Mat','C_Mat','-v7.3');
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated and saved Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        elseif SaveCalMat == 0
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        else
                            fprintf('Warning. SaveCalMat not properly defined. Defaulting to Disabled.\n')
                            SaveCalMat = 0;
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask, Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        end
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat*(M_Mod_v);
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        elseif Use_CalMat == 1
            path_mat = pwd;
            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1
                        load(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','C_Mat','R_Mat')
                        Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                        clear('C_Mat','R_Mat');
                    else
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat*(M_Mod_v);
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        else
            fprintf('Error. Use_CalMat must be 1(activated) or 0(deactivated).\n')
            return
        end
    elseif LC_Flag == 1
        %Determination of the maximum angle projection.
        max_Ang = max(Tilt_v);
        min_Ang = min(Tilt_v);
        if max_Ang >= abs(min_Ang) && max_Ang < 75 && min_Ang > -75
        elseif max_Ang < abs(min_Ang) && max_Ang < 75 && min_Ang > -75
            max_Ang = abs(min_Ang);
        else
            fprintf('Error. LC_Flag is active but the maximum projection angle is larger than 75 deg.\n')
            fprintf('The maximum of the projection angles must be between -75 to 75 degrees.\n')
            return
        end
        ext_Mod = ceil(Mod_Size(3)*tand(max_Ang));
        %Model Initialization
        if strcmp(Recon_Flag,'XTiltVector') == 1
            M_Mod_v = zeros(2*Mod_Size(1)*(Mod_Size(2)+2*ext_Mod)*Mod_Size(3),1);
        elseif strcmp(Recon_Flag,'YTiltVector') == 1
            M_Mod_v = zeros(2*(Mod_Size(1)+2*ext_Mod)*Mod_Size(2)*Mod_Size(3),1);
        else
        end
        if Use_CalMat == 0
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1 %Saving the projection matrices in the HDD to save execution time.
                        if slc == 1 && SaveCalMat == 1
                            mkdir('Mag_Projection_Matrices')
                        else
                        end
                        if SaveCalMat == 1
                            path_mat = pwd;
                            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            save(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','R_Mat','C_Mat','-v7.3');
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated and saved Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        elseif SaveCalMat == 0
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        else
                            fprintf('Warning. SaveCalMat not properly defined. Defaulting to Disabled.\n')
                            SaveCalMat = 0;
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        end
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat*(M_Mod_v);
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        elseif Use_CalMat == 1
            path_mat = pwd;
            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1
                        load(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','C_Mat','R_Mat')
                        Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                        clear('C_Mat','R_Mat');
                    else
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat*(M_Mod_v);
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        else
            fprintf('Error. Use_CalMat must be 1(activated) or 0(deactivated).\n')
            return
        end
    else
        fprintf('Error. LC_Flag must be 1(activated) or 0(deactivated).\n')
        return
    end
elseif GPU_Flag == 1 %GPU Implementation
    M_Tomo_v = gpuArray(M_Tomo_v);
    if LC_Flag == 0
        ext_Mod = 0;
        %Models Initialization
        M_Mod_v = gpuArray(zeros(2*Mod_Size(1)*Mod_Size(2)*Mod_Size(3),1));
        if Use_CalMat == 0
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1 %Saving the projection matrices in the HDD to save execution time.
                        if slc == 1 && SaveCalMat == 1
                            mkdir('Mag_Projection_Matrices')
                        else
                        end
                        if SaveCalMat == 1
                            path_mat = pwd;
                            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            save(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','R_Mat','C_Mat','-v7.3');
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated and saved Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        elseif SaveCalMat == 0
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        else
                            fprintf('Warning. SaveCalMat not properly defined. Defaulting to Disabled.')
                            SaveCalMat = 0;
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        end
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat_GPU*M_Mod_v;
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat_GPU*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        elseif Use_CalMat == 1
            path_mat = pwd;
            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1
                        load(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','C_Mat','R_Mat')
                        Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                        clear('C_Mat','R_Mat');
                        Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                        Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat_GPU*M_Mod_v;
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat_GPU*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        else
            fprintf('Error. Use_CalMat must be 1(activated) or 0(deactivated).\n')
            return
        end
    elseif LC_Flag == 1
        %Determination of the maximum angle projection.
        max_Ang = max(Tilt_v);
        min_Ang = min(Tilt_v);
        if max_Ang >= abs(min_Ang) && max_Ang < 75 && min_Ang > -75
        elseif max_Ang < abs(min_Ang) && max_Ang < 75 && min_Ang > -75
            max_Ang = abs(min_Ang);
        else
            fprintf('Error. LC_Flag is active but the maximum projection angle is larger than 75 deg.\n')
            fprintf('The maximum of the projection angles must be between -75 to 75 degrees.\n')
            return
        end
        ext_Mod = ceil(Mod_Size(3)*tand(max_Ang));
        %Model Initialization
        if strcmp(Recon_Flag,'XTiltVector') == 1
            M_Mod_v = gpuArray(zeros(2*Mod_Size(1)*(Mod_Size(2)+2*ext_Mod)*Mod_Size(3),1));
        elseif strcmp(Recon_Flag,'YTiltVector') == 1
            M_Mod_v = gpuArray(zeros(2*(Mod_Size(1)+2*ext_Mod)*Mod_Size(2)*Mod_Size(3),1));
        else
        end
        if Use_CalMat == 0
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1 %Saving the projection matrices in the HDD to save execution time.
                        if slc == 1 && SaveCalMat == 1
                            mkdir('Mag_Projection_Matrices')
                        else
                        end
                        if SaveCalMat == 1
                            path_mat = pwd;
                            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            save(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','R_Mat','C_Mat','-v7.3');
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated and saved Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        elseif SaveCalMat == 0
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        else
                            fprintf('Warning. SaveCalMat not properly defined. Defaulting to Disabled.')
                            SaveCalMat = 0;
                            [Fwd_Proj_Mat,R_Mat,C_Mat] = VecTomo_Fwd_Mat_Parallel_SingTiltSIRT(Recon_Mask,Det_Size,[Fast_Idx,slc],Pix_Size,Mod_Size,Tilt_v,Recon_Flag,SimultSlcs,LC_Flag,max_Ang);
                            Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                            Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                            Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                            clear('C_Mat','R_Mat');
                            fprintf('Calculated Reconstruction Matrices of Slice %i of %i.\n',slc,Slice_Count)
                        end
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat_GPU*M_Mod_v;
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat_GPU*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        elseif Use_CalMat == 1
            path_mat = pwd;
            path_mat = [path_mat,'\Mag_Projection_Matrices\'];
            %---------------------
            %Vector Reconstruction
            %---------------------
            for slc=1:Slice_Count
                for it=1:N_iter
                    if it == 1
                        load(sprintf('%sSlice_%i.mat',path_mat,slc),'Fwd_Proj_Mat','C_Mat','R_Mat')
                        Bkwd_Proj_Mat = C_Mat*Fwd_Proj_Mat'*R_Mat;
                        clear('C_Mat','R_Mat');
                        Fwd_Proj_Mat_GPU = gpuArray(Fwd_Proj_Mat);
                        Bkwd_Proj_Mat_GPU = gpuArray(Bkwd_Proj_Mat);
                    else
                        %Do nothing
                    end
                    err_v = M_Tomo_v(:,slc)-Fwd_Proj_Mat_GPU*M_Mod_v;
                    M_Mod_v = M_Mod_v + Bkwd_Proj_Mat_GPU*err_v;
                    fprintf('Slice %i of %i. Completed Iteration %i of %i.\n',slc,Slice_Count,it,N_iter)
                end
            end
            %Saving Temporary Data
            N_iter_comp = N_iter;
            save('Temp_Recon.mat','M_Mod_v','Mod_Size','ext_Mod','Recon_Flag','SimultSlcs','N_iter_comp');
        else
            fprintf('Error. Use_CalMat must be 1(activated) or 0(deactivated).\n')
            return
        end
    else
        fprintf('Error. LC_Flag must be 1(activated) or 0(deactivated).\n')
        return
    end
    M_Mod_v = gather(M_Mod_v);
else
    fprintf('ERROR\nGPU_Flag must be 0 or 1.\n')
    return
end
%Output Arrangement.
if LC_Flag == 0
    m1_Mod = zeros(Mod_Size(2),Mod_Size(1),Mod_Size(3));
    m2_Mod = zeros(Mod_Size(2),Mod_Size(1),Mod_Size(3));
    for k=1:Mod_Size(3)
        for i=1:Mod_Size(2)
            m1_Mod(i,:,k) = M_Mod_v(1+(i-1)*Mod_Size(1)+(k-1)*Mod_Size(1)*Mod_Size(2):...
                Mod_Size(1)+(i-1)*Mod_Size(1)+(k-1)*Mod_Size(1)*Mod_Size(2))';
            m2_Mod(i,:,k) = M_Mod_v(Mod_Size(1)*Mod_Size(2)*Mod_Size(3)+1+(i-1)*Mod_Size(1)+(k-1)*Mod_Size(1)*Mod_Size(2):...
                Mod_Size(1)*Mod_Size(2)*Mod_Size(3)+Mod_Size(1)+(i-1)*Mod_Size(1)+(k-1)*Mod_Size(1)*Mod_Size(2))';
        end
    end
elseif LC_Flag == 1
    if strcmp(Recon_Flag,'XTiltVector') == 1
        Mod_SizeAr = [Mod_Size(1),Mod_Size(2)+2*ext_Mod,Mod_Size(3)];
        m1_Mod = zeros(Mod_SizeAr(2),Mod_SizeAr(1),Mod_SizeAr(3));
        m2_Mod = zeros(Mod_SizeAr(2),Mod_SizeAr(1),Mod_SizeAr(3));
    elseif strcmp(Recon_Flag,'YTiltVector') == 1
        Mod_SizeAr = [Mod_Size(1)+2*ext_Mod,Mod_Size(2),Mod_Size(3)];
        m1_Mod = zeros(Mod_SizeAr(2),Mod_SizeAr(1),Mod_SizeAr(3));
        m2_Mod = zeros(Mod_SizeAr(2),Mod_SizeAr(1),Mod_SizeAr(3));
    else
    end
    for k=1:Mod_SizeAr(3)
        for i=1:Mod_SizeAr(2)
            m1_Mod(i,:,k) = M_Mod_v(1+(i-1)*Mod_SizeAr(1)+(k-1)*Mod_SizeAr(1)*Mod_SizeAr(2):...
                Mod_SizeAr(1)+(i-1)*Mod_SizeAr(1)+(k-1)*Mod_SizeAr(1)*Mod_SizeAr(2))';
            m2_Mod(i,:,k) = M_Mod_v(Mod_SizeAr(1)*Mod_SizeAr(2)*Mod_SizeAr(3)+1+(i-1)*Mod_SizeAr(1)+(k-1)*Mod_SizeAr(1)*Mod_SizeAr(2):...
                Mod_SizeAr(1)*Mod_SizeAr(2)*Mod_SizeAr(3)+Mod_SizeAr(1)+(i-1)*Mod_SizeAr(1)+(k-1)*Mod_SizeAr(1)*Mod_SizeAr(2))';
        end
    end
    if strcmp(Recon_Flag,'XTiltVector') == 1
        m1_Mod = m1_Mod(ext_Mod+1:Mod_Size(2)+ext_Mod,:,:);
        m2_Mod = m2_Mod(ext_Mod+1:Mod_Size(2)+ext_Mod,:,:);
    elseif strcmp(Recon_Flag,'YTiltVector') == 1
        m1_Mod = m1_Mod(:,ext_Mod+1:Mod_Size(1)+ext_Mod,:);
        m2_Mod = m2_Mod(:,ext_Mod+1:Mod_Size(1)+ext_Mod,:);
    else
    end
else
    fprintf('Error. LC_Flag must be 1(activated) or 0(deactivated).\n')
    return
end
end