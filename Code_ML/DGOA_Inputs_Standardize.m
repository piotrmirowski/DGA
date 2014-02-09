% DGOA_Inputs_Standardize  Normalize data to zero mean, unit variance
%
% Syntax:
%   [x_pos, x_neg, x_m, x_s] = DGOA_Inputs_Standardize(x_pos, x_neg)
% Inputs:
%   x_pos: matrix of size <D> x <M> of positive input data
%   x_neg: matrix of size <D> x <N> of negative input data
% Outputs:
%   x_pos: matrix of size <D> x <M> of normalized positive input data
%   x_neg: matrix of size <D> x <N> of normalized negative input data
%   x_m:   vector of length <D> of means
%   x_s:   vector of length <D> of standard deviations

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
% Version 1.0, New York, 27 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [x_pos, x_neg, x_m, x_s] = DGOA_Inputs_Standardize(x_pos, x_neg)

% Compute the mean and standard deviation vectors, for each variable
x_m = mean([x_pos x_neg]')';
x_s = std([x_pos x_neg]')';

% Normalize the positive and negative data to unit variance and zero mean
len_pos = size(x_pos, 2);
len_neg = size(x_neg, 2);
x_pos = (x_pos - repmat(x_m, 1, len_pos)) ./ repmat(x_s, 1, len_pos);
if (len_neg > 0)
  x_neg = (x_neg - repmat(x_m, 1, len_neg)) ./ repmat(x_s, 1, len_neg);
else
  x_neg = [];
end