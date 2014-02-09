% DGOA_InverseCDF  Inverse CDF function
%
% Syntax:
%   f = DGOA_InverseCDF(x, xCDF, fCDF)
% Inputs:
%   x:    scalar or vector at which the CDF is evaluated
%   xCDF: vector of sorted input values
%   fCDF: vector of CDF values associated to <xCDF>
% Outputs:
%   y:    vector of interpolated CDF values

% Copyright (C) 2009 Piotr Mirowski
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
% Version 1.0, New York, 08 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function f = DGOA_InverseCDF(x, xCDF, fCDF)

% Check that the x of the CDF are unique
[xCDF, ind] = unique(xCDF);
fCDF = fCDF(ind);

% Interpolate using the CDF
f = interp1(xCDF, fCDF, x);

% Set values outside of the bounds of <xCDF> to 0 or 1
f(x < min(xCDF)) = 0;
f(x > max(xCDF)) = 1;
