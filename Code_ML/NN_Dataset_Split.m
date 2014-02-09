% NN_Dataset_Split  Initialize the learning and cross-validation datasets
%
% Syntax:
%   [datasetLearn, datasetXval] = NN_Dataset_Split(x, y, params)
% Inputs:
%   x:            matrix of size <dim_x> x <n> of inputs
%   y:            matrix of size <dim_y> x <n> of target outputs
%   params:  struct containing the parameters
% Outputs:
%   datasetLearn: struct containing the dataset used for training
%   datasetXval:  struct containing the dataset used for cross-validation

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

function [datasetLearn, datasetXval] = NN_Dataset_Split(x, y, params)

% Split the data into learning and cross-validation
n = size(x, 2);
ind = randperm(n);
if ((params.perc_xval < 0) || (params.perc_xval > 1))
  error('<params.perc_xval> needs to be between 0 and 1');
end
n_learn = round(n * (1 - params.perc_xval));
indLearn = ind(1:n_learn);
indXval = ind((n_learn+1):end);

% Sanity check
if (n ~= size(y, 2))
  error('<x> and <y> need to have the same number of columns (samples)');
end

% Learning dataset
datasetLearn = struct('n', n_learn, 'len_batch', params.len_batch);
datasetLearn.x = x(:, indLearn);
datasetLearn.y = double(y(:, indLearn));

% Learning dataset
datasetXval = struct('n', n - n_learn, 'len_batch', params.len_batch);
datasetXval.x = x(:, indXval);
datasetXval.y = double(y(:, indXval));
