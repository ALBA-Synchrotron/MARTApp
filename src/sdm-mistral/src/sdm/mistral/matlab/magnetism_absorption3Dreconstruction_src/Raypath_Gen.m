%Function to calculate the line path of any probe through a 3D
%voxel volume model for tomographic reconstruction purposes. The input are
%the pixel positions of the detector, the rotation axis related with the
%detector, and the rotation angle. The rotation axis is determined by a
%vector indicating the rotation axis. The outputs are the interaction
%voxels concerning each specific pixel given by their natural indices in
%the 3D tensor and the length of each ray through each specific voxel.
%--------------------------------------------------------------------------
function [x_idx,y_idx,z_idx,len_v,P_det,P_sour] = Raypath_Gen(Det_pos_x,Det_pos_y,...
    Rot_Axis,Rot_Offset_Pos,Tilt_Ang,Mod_Size,Vox_Size,Mod_Rot_Axis,...
    Mod_Rot_Offset_Pos,Mod_Rot_Ang)
%--------------------------------------------------------------------------
%Function Inputs
%
%   Det_pos_x:      number indicating the x position of the detector
%                   corresponding to the pixel to be considered for the
%                   calculation of the ray path through the volume model.
%   Det_pos_y:      number indicating the y position of the detector
%                   corresponding to the pixel to be considered for the
%                   calculation of the ray path through the volume model.
%   Rot_Axis:       three elements vector containing the direction and
%                   sense of the rotation axis. The three components
%                   correspond to the [x,y,z] components of the
%                   aforementioned vector. The reference frame is refered
%                   to the detector itself thus X corresponds to the
%                   horizontal direction or the columns in the detector
%                   and Y corresponds to the vertical direction or the rows
%                   in the detector. The origin is located at the center of
%                   the voxels volume model which is by default in front of
%                   the detector. A modification of this position is
%                   possible by using the Offset_Pos input variable.
%                   Ex: Rotation around Y axis [0,1,0].
%   Rot_Offset_Pos: three elements vector containing the offset between the
%                   origin (and the rotation axis which passes by it), and
%                   the center of the volume voxels model. The elements
%                   indicate the offset [x_off,y_off,z_off] in the three
%                   components of the cartesian three-dimensional space.
%   Tilt_Ang:       Rotation angle in degrees for the specific ray path to
%                   be calculated. This rotation is applied around the
%                   rotation axis as defined by the combination of the
%                   inputs from variables Rot_Axis and Offset_Pos.
%   Mod_Size:       three elements vector indicating the size of the volume
%                   voxels model. The size along X, Y and Z directions are
%                   indicated in order. [Mod_sizeX,Mod_sizeY,Mod_sizeZ].
%                   These numbers correspond to the number of voxels along
%                   each direction, not to the total length of the model in
%                   real units.
%   Vox_Size:       three elements vector conaining the size of each voxel
%                   of the model. The units are related to the pixels of
%                   the detector. This means that a cell size of [1,1,1] is
%                   taking a cubic voxel with the same size of a pixel in
%                   the detector. This allows for automatic upsampling and
%                   downsampling of the output form the general
%                   reconstruction algorithm.
%   Mod_Rot_Axis:   three elements vector containing the direction and
%                   sense of the rotation axis of the model. The three
%                   components correspond to the [x,y,z] components of the
%                   aforementioned vector. The reference frame is refered
%                   to the model itself. The origin is located at the
%                   center of the voxels volume model. A modification of
%                   this position is possible by using the Mod_Offset_Pos
%                   input variable. This is usefull for double tilt
%                   tomography measurements in order to use during the
%                   reconstruction always the same volume voxels model.
%                   Ex: Rotation around Y axis of the Model [0,1,0].
%   Mod_Rot_Offset_Pos: three components vector containing the offset from
%                       the rotation axis position, to the center of the
%                       volume voxels model in voxel units.
%   Mod_Rot_Ang:    Rotation angle of the model around the model rotation
%                   axis.
%
%--------------------------------------------------------------------------
%Function Outputs
%
%   x_idx:          Column vector containing the along x indices of the
%                   voxels in the volume model interacting with the
%                   specific ray-path calculated. The along x indices
%                   correspond to the column dimension of the
%                   three-dimensional tensor of the model.[Matlab's 2nd indx]
%   y_idx:          Column vector containing the along y indices of the
%                   voxels in the volume model interacting with the
%                   specific ray-path calculated. The along y indices
%                   correspond to the rows dimension of the
%                   three-dimensional tensor of the model.[Matlab's 1st indx]
%   z_idx:          Column vector containing the along z indices of the
%                   voxels in the volume model interacting with the
%                   specific ray-path calculated. The along z indices
%                   correspond to the slices dimension of the
%                   three-dimensional tensor of the model.[Matlab's 3rd indx]
%   len_v:          Column vector containing the length of the ray through
%                   each of the voxels indicated by the x_idx,y_idx,z_idx
%                   column vectors. The units of this length are in natural
%                   voxel units.
%   P_det:          For debugging. Coordinates X,Y,Z of the detector in the
%                   microscope frame of reference.
%   P_sour:         For debugging. Coordinates X,Y,Z of the source in the
%                   microscope frame of reference.
%
%--------------------------------------------------------------------------
% Code created by Aurelio Hierro Rodriguez at University of Oviedo (Spain)
% hierroaurelio@uniovi.es/aurehr2001@gmail.com
% 21/03/2020
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
%Defining the initial Ray From Source To Detector
P_det = [Det_pos_x;Det_pos_y;3*max(Mod_Size)]; %Considering Positive the Z direction towards the detector.
P_sour = [Det_pos_x;Det_pos_y;-3*max(Mod_Size)];
%First orienting the volume model respect to its rotation direction and
%twist angle. The rotation of the voulme equals the -rotation of the beam.
Rot_mod = RotMatGen(Mod_Rot_Axis,-Mod_Rot_Ang);
P_det = Rot_mod*P_det;
P_sour = Rot_mod*P_sour;
%Second -rotation of the beam to mimic the tilt of the model during the
%tomography experiment.
Transf_Rot_Axis = Rot_mod*Rot_Axis';
Rot_tilt = RotMatGen(Transf_Rot_Axis',-Tilt_Ang);
P_det = Rot_tilt*P_det;
P_sour = Rot_tilt*P_sour;
%Calculation of the Model offsets respect to the tilt rotation axis position.
%We are going to displace the model the separation between both rotation axes.
bx = -Mod_Size(1)/2*Vox_Size(1)+Rot_Offset_Pos(1)+Mod_Rot_Offset_Pos(1);
by = -Mod_Size(2)/2*Vox_Size(2)+Rot_Offset_Pos(2)+Mod_Rot_Offset_Pos(2);
bz = -Mod_Size(3)/2*Vox_Size(3)+Rot_Offset_Pos(3)+Mod_Rot_Offset_Pos(3);
%Intersection with the borders of the Model Space.
alph_xmin = min([(bx-P_sour(1))/(P_det(1)-P_sour(1)),((bx+Mod_Size(1)*Vox_Size(1))-P_sour(1))/(P_det(1)-P_sour(1))]);
alph_xmax = max([(bx-P_sour(1))/(P_det(1)-P_sour(1)),((bx+Mod_Size(1)*Vox_Size(1))-P_sour(1))/(P_det(1)-P_sour(1))]);
alph_ymin = min([(by-P_sour(2))/(P_det(2)-P_sour(2)),((by+Mod_Size(2)*Vox_Size(2))-P_sour(2))/(P_det(2)-P_sour(2))]);
alph_ymax = max([(by-P_sour(2))/(P_det(2)-P_sour(2)),((by+Mod_Size(2)*Vox_Size(2))-P_sour(2))/(P_det(2)-P_sour(2))]);
alph_zmin = min([(bz-P_sour(3))/(P_det(3)-P_sour(3)),((bz+Mod_Size(3)*Vox_Size(3))-P_sour(3))/(P_det(3)-P_sour(3))]);
alph_zmax = max([(bz-P_sour(3))/(P_det(3)-P_sour(3)),((bz+Mod_Size(3)*Vox_Size(3))-P_sour(3))/(P_det(3)-P_sour(3))]);
alph_min = max([alph_xmin,alph_ymin,alph_zmin]);
alph_max = min([alph_xmax,alph_ymax,alph_zmax]);
if alph_min >= alph_max
    %Ray do not intercat with the model volume.
    len_v = [];
    x_idx = [];
    y_idx = [];
    z_idx = [];
    return
else
    if P_sour(1) < P_det(1) || abs(P_sour(1)-P_det(1)) < 1e-13 %Determination of the intersected first and last X planes after the beam enters the model space.
        if alph_min == alph_xmin
            imin = 1;
        else
            imin = ceil((P_sour(1)+alph_min*(P_det(1)-P_sour(1))-bx)/Vox_Size(1));
        end
        if alph_max == alph_xmax
            imax = Mod_Size(1);
        else
            imax = floor((P_sour(1)+alph_max*(P_det(1)-P_sour(1))-bx)/Vox_Size(1));
        end
        alph_x = (bx+(imin)*Vox_Size(1)-P_sour(1))/(P_det(1)-P_sour(1));
    else
        if alph_min == alph_xmin
            imax = Mod_Size(1)-1;
        else
            imax = floor((P_sour(1)+alph_min*(P_det(1)-P_sour(1))-bx)/Vox_Size(1));
        end
        if alph_max == alph_xmax
            imin = 0;
        else
            imin = ceil((P_sour(1)+alph_max*(P_det(1)-P_sour(1))-bx)/Vox_Size(1));
        end
        alph_x = (bx+(imax)*Vox_Size(1)-P_sour(1))/(P_det(1)-P_sour(1));
    end
    if P_sour(2) < P_det(2) || abs(P_sour(2)-P_det(2)) < 1e-13 %Determination of the intersected first and last Y planes after the beam enters the model space.
        if alph_min == alph_ymin
            jmin = 1;
        else
            jmin = ceil((P_sour(2)+alph_min*(P_det(2)-P_sour(2))-by)/Vox_Size(2));
        end
        if alph_max == alph_ymax
            jmax = Mod_Size(2);
        else
            jmax = floor((P_sour(2)+alph_max*(P_det(2)-P_sour(2))-by)/Vox_Size(2));
        end
        alph_y = (by+(jmin)*Vox_Size(2)-P_sour(2))/(P_det(2)-P_sour(2));
    else
        if alph_min == alph_ymin
            jmax = Mod_Size(2)-1;
        else
            jmax = floor((P_sour(2)+alph_min*(P_det(2)-P_sour(2))-by)/Vox_Size(2));
        end
        if alph_max == alph_ymax
            jmin = 0;
        else
            jmin = ceil((P_sour(2)+alph_max*(P_det(2)-P_sour(2))-by)/Vox_Size(2));
        end
        alph_y = (by+(jmax)*Vox_Size(2)-P_sour(2))/(P_det(2)-P_sour(2));
    end
    if P_sour(3) < P_det(3) || abs(P_sour(3)-P_det(3)) < 1e-13 %Determination of the intersected first and last Z planes after the beam enters the model space.
        if alph_min == alph_zmin
            kmin = 1;
        else
            kmin = ceil((P_sour(3)+alph_min*(P_det(3)-P_sour(3))-bz)/Vox_Size(3));
        end
        if alph_max == alph_zmax
            kmax = Mod_Size(3);
        else
            kmax = floor((P_sour(3)+alph_max*(P_det(3)-P_sour(3))-bz)/Vox_Size(3));
        end
        alph_z = (bz+(kmin)*Vox_Size(3)-P_sour(3))/(P_det(3)-P_sour(3));
    else
        if alph_min == alph_zmin
            kmax = Mod_Size(3)-1;
        else
            kmax = floor((P_sour(3)+alph_min*(P_det(3)-P_sour(3))-bz)/Vox_Size(3));
        end
        if alph_max == alph_zmax
            kmin = 0;
        else
            kmin = ceil((P_sour(3)+alph_max*(P_det(3)-P_sour(3))-bz)/Vox_Size(3));
        end
        alph_z = (bz+(kmax)*Vox_Size(3)-P_sour(3))/(P_det(3)-P_sour(3));
    end
    %Rays Normal to XY plane
    if abs(P_sour(1)-P_det(1)) < 1e-13 && abs(P_sour(2)-P_det(2)) < 1e-13
        arg = (alph_z+alph_min)/2;
        %Rays Parallel to XY plane
    elseif abs(P_sour(3)-P_det(3)) < 1e-13
        arg = (min([alph_x,alph_y])+alph_min)/2;
        %Rays Normal to XZ plane
    elseif abs(P_sour(1)-P_det(1)) < 1e-13 && abs(P_sour(3)-P_det(3)) < 1e-13
        arg = (alph_y+alph_min)/2;
        %Rays Parallel to XZ plane
    elseif abs(P_sour(2)-P_det(2)) < 1e-13
        arg = (min([alph_x,alph_z])+alph_min)/2;
        %Rays Normal to YZ plane
    elseif abs(P_sour(2)-P_det(2)) < 1e-13 && abs(P_sour(3)-P_det(3)) < 1e-13
        arg = (alph_x+alph_min)/2;
        %Rays Parallel to YZ plane
    elseif abs(P_sour(1)-P_det(1)) < 1e-13
        arg = (min([alph_y,alph_z])+alph_min)/2;
        %Generic Ray
    else
        arg = (min([alph_x,alph_y,alph_z])+alph_min)/2;
    end
    i1 = floor((P_sour(1)+arg*(P_det(1)-P_sour(1))-bx)/Vox_Size(1));
    j1 = floor((P_sour(2)+arg*(P_det(2)-P_sour(2))-by)/Vox_Size(2));
    k1 = floor((P_sour(3)+arg*(P_det(3)-P_sour(3))-bz)/Vox_Size(3));
    alph_xu =Vox_Size(1)/abs(P_det(1)-P_sour(1));
    alph_yu =Vox_Size(2)/abs(P_det(2)-P_sour(2));
    alph_zu =Vox_Size(3)/abs(P_det(3)-P_sour(3));
    if P_sour(1) < P_det(1)
        iu = 1;
    else
        iu = -1;
    end
    if P_sour(2) < P_det(2)
        ju = 1;
    else
        ju = -1;
    end
    if P_sour(3) < P_det(3)
        ku = 1;
    else
        ku = -1;
    end
    alph_c = alph_min;
    Np = imax-imin+1+jmax-jmin+1+kmax-kmin+1;
    len_v = zeros(Np,1);
    x_idx = zeros(Np,1);
    y_idx = zeros(Np,1);
    z_idx = zeros(Np,1);
    dtot = sqrt(sum((P_det-P_sour).^2,1));
    for cc = 1:Np
        if alph_x < 0 || alph_x > 1 % Ray in plane YZ
            if alph_y < alph_z
                len_v(cc) = (alph_y-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                j1 = j1+ju;
                alph_c = alph_y;
                alph_y = alph_y+alph_yu;
            else
                len_v(cc) = (alph_z-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                k1 = k1+ku;
                alph_c = alph_z;
                alph_z = alph_z+alph_zu;
            end
        elseif alph_y < 0 || alph_y > 1 % Ray in plane XZ
            if alph_x < alph_z
                len_v(cc) = (alph_x-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                i1 = i1+iu;
                alph_c = alph_x;
                alph_x = alph_x+alph_xu;
            else
                len_v(cc) = (alph_z-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                k1 = k1+ku;
                alph_c = alph_z;
                alph_z = alph_z+alph_zu;
            end
        elseif alph_z < 0 || alph_z > 1 % Ray in plane XY
            if alph_x < alph_y
                len_v(cc) = (alph_x-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                i1 = i1+iu;
                alph_c = alph_x;
                alph_x = alph_x+alph_xu;
            else
                len_v(cc) = (alph_y-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                j1 = j1+ju;
                alph_c = alph_y;
                alph_y = alph_y+alph_yu;
            end
        else
            if alph_x < alph_y && alph_x < alph_z
                len_v(cc) = (alph_x-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                i1 = i1+iu;
                alph_c = alph_x;
                alph_x = alph_x+alph_xu;
            elseif alph_y < alph_x && alph_y < alph_z
                len_v(cc) = (alph_y-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                j1 = j1+ju;
                alph_c = alph_y;
                alph_y = alph_y+alph_yu;
            else
                len_v(cc) = (alph_z-alph_c)*dtot;
                x_idx(cc) = i1;
                y_idx(cc) = j1;
                z_idx(cc) = k1;
                k1 = k1+ku;
                alph_c = alph_z;
                alph_z = alph_z+alph_zu;
            end
        end
    end
    [i] = find(len_v == 0);
    if max(i) == Np
        len_v = len_v(1:Np-1);
        x_idx = x_idx(1:Np-1)+1;
        y_idx = y_idx(1:Np-1)+1;
        z_idx = z_idx(1:Np-1)+1;
    else
        x_idx = x_idx+1;
        y_idx = y_idx+1;
        z_idx = z_idx+1;
    end
end
end
