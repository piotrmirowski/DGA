% ML_SplitLearnXval  Split a dataset into 5-fold learning and x-val sets
%
% Syntax:
%   [xLearn, yLearn, xXval, yXval] = ML_SplitLearnXval(xTrain, yTrain)
% Inputs:
%   xTrain: matrix of size <dim_x> x <n> of inputs
%   yTrain: matrix of size <dim_y> x <n> of target outputs
% Outputs:
%   xLearn: cell array of 5 matrices of size <dim_x> x <4*n/5> of inputs
%   yLearn: cell array of 5 matrices of size <dim_y> x <4*n/5> of targets
%   xXval:  cell array of 5 matrices of size <dim_x> x <n/5> of inputs
%   yXval:  cell array of 5 matrices of size <dim_y> x <n/5> of targets

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
% Version 1.0, New York, 18 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [xLearn, yLearn, xXval, yXval] = ML_SplitLearnXval(xTrain, yTrain)

% Define the cross-validation and learning sets
n_samples = size(yTrain, 2);
n_samples_5 = ceil(n_samples / 5);
indXval = cell(1, 5);
indLearn = cell(1, 5);
xXval = cell(1, 5);
xLearn = cell(1, 5);
yXval = cell(1, 5);
yLearn = cell(1, 5);
for k = 1:5
  indXval{k} = (n_samples_5*(k-1)+1):(n_samples_5*k);
  indXval{k} = unique(min(n_samples, indXval{k}));
  indLearn{k} = setdiff(1:n_samples, indXval{k});
  xLearn{k} = xTrain(:, indLearn{k});
  xXval{k} = xTrain(:, indXval{k});
  yLearn{k} = yTrain(:, indLearn{k});
  yXval{k} = yTrain(:, indXval{k});
end
