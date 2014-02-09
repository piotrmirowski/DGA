% NN_Dataset_ForwardProp  Get the <k>-th mini-batch from the dataset
%
% Syntax:
%   [x, y, ind] = NN_Dataset_ForwardProp(dataset, params, k)
% Inputs:
%   dataset: struct containing the dataset used for training/evaluation
%   params:  struct containing the parameters
%   k:       scalar, number of the mini-batch to be retrieved
% Outputs:
%   x:       matrix of size <dim_x> x <n_k> of inputs
%   y:       matrix of size <dim_y> x <n_k> of target outputs
%   ind:     vector of length <n_k> of sample indexes

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

function [x, y, ind] = NN_Dataset_ForwardProp(dataset, k)

k0 = (k - 1) * dataset.len_batch + 1;
k1 = k * dataset.len_batch;
ind = unique(min([k0:k1], dataset.n));

% Try using a scrambled order of datapoints
try
  ind = dataset.order(ind);
end

% Select the minibatch
y = dataset.y(:, ind);
x = dataset.x(:, ind);

