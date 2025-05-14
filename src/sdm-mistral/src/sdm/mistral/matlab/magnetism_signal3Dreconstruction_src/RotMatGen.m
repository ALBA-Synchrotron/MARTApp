%Function to create a rotation matrix around a generic rotation axis.
%--------------------------------------------------------------------------
function [Rot_mat] = RotMatGen(Rot_Axis,Rot_ang)
%--------------------------------------------------------------------------
% Function Input
%
%   Rot_Axis:   three elements vector indicating the direction of the
%               rotation axis in the cartesian coordinates system aligned
%               with the detector. Ex: [0,1,0] rotation around Y direction.
%   Rot_Ang:    Rotation Angle in degrees to perform the tilt.
%
%--------------------------------------------------------------------------
% Function Output
%
%   Rot_mat:    three by three matrix general rotation matrix from the
%               information in the input arguments.
%
%--------------------------------------------------------------------------
% Code created by Aurelio Hierro Rodriguez at University of Oviedo (Spain)
% hierroaurelio@uniovi.es/aurehr2001@gmail.com
% 22/03/2020
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
%Converting the Angle to radians
Rot_ang = Rot_ang*pi/180;
%Normalization of the Rotation Axis Vector
Rot_Axis = Rot_Axis./((sum(Rot_Axis.^2))^(1/2));
%Getting the Euler angles for the General Rotation Axis
thet = acos(Rot_Axis(3));
if Rot_Axis(1) == 0 && Rot_Axis(2) == 0 || Rot_Axis(1) == 1
    phi = 0;
elseif Rot_Axis(1) >= 0 && Rot_Axis(2) > 0
    phi = atan(Rot_Axis(2)/Rot_Axis(1));
elseif Rot_Axis(1) >= 0 && Rot_Axis(2) < 0
    phi = 2*pi + atan(Rot_Axis(2)/Rot_Axis(1));
elseif Rot_Axis(1) < 0 && Rot_Axis(2) > 0
    phi = pi + atan(Rot_Axis(2)/Rot_Axis(1));
elseif Rot_Axis(1) < 0 && Rot_Axis(2) < 0
    phi = pi + atan(Rot_Axis(2)/Rot_Axis(1));
else
    fprintf('ERROR in RotMatGen function.\n')
    return
end
%Creating the unitary rotation matrices
cos_Rot = cos(Rot_ang);
sin_Rot = sin(Rot_ang);
cos_Phi = cos(phi);
sin_Phi = sin(phi);
cos_Thet = cos(thet);
sin_Thet = sin(thet);

Rz_pos = [cos_Phi,-sin_Phi,0;sin_Phi,cos_Phi,0;0,0,1];
Ry_pos = [cos_Thet,0,sin_Thet;0,1,0;-sin_Thet,0,cos_Thet];
Rz_rot = [cos_Rot,-sin_Rot,0;sin_Rot,cos_Rot,0;0,0,1];
Rz_neg = [cos_Phi,sin_Phi,0;-sin_Phi,cos_Phi,0;0,0,1];
Ry_neg = [cos_Thet,0,-sin_Thet;0,1,0;sin_Thet,0,cos_Thet];
%Constructing the final rotation matrix
Rot_mat = Rz_pos*Ry_pos*Rz_rot*Ry_neg*Rz_neg;
end
