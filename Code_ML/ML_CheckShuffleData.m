% ML_CheckShuffleData  Check data format and shuffle train datapoints
%
% Syntax:
%   [xTrain, yTrain, xTest, yTest, xTrain_copy, yTrain_copy] = ...
%     ML_CheckShuffleData(xTrain, yTrain, xTest, yTest)
% Inputs:
%   xTrain:      matrix of size <dim_x> x <n1> of inputs
%   yTrain:      matrix of size <dim_y> x <n1> of target outputs
%   xTest:       matrix of size <dim_x> x <n2> of inputs
%   yTest:       matrix of size <dim_y> x <n2> of target outputs
% Outputs:
%   xTrain:      matrix of size <dim_x> x <n1> of shuffled inputs
%   yTrain:      matrix of size <dim_y> x <n1> of shuffled target outputs
%   xTest:       matrix of size <dim_x> x <n2> of inputs
%   yTest:       matrix of size <dim_y> x <n2> of target outputs
%   xTrain_copy: matrix of size <dim_x> x <n1> of unshuffled inputs
%   yTrain_copy: matrix of size <dim_y> x <n1> of unshuffled target outputs

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
% Version 1.0, New York, 24 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [xTrain, yTrain, xTest, yTest, xTrain_copy, yTrain_copy] = ...
  ML_CheckShuffleData(xTrain, yTrain, xTest, yTest)

% Format the data properly
n_train = length(yTrain);
yTrain = logical(yTrain);
if (size(yTrain, 2) == 1)
  yTrain = yTrain';
end
if (size(xTrain, 2) ~= n_train)
  xTrain = xTrain';
  if (size(xTrain, 2) ~= n_train)
    error('The size of xTrain does not match the size of yTrain');
  end
end
n_test = length(yTest);
yTest = logical(yTest);
if (size(yTest, 2) == 1)
  yTest = yTest';
end
if (size(xTest, 2) ~= n_test)
  xTest = xTest';
  if (size(xTest, 2) ~= n_test)
    error('The size of xTest does not match the size of yTest');
  end
end

% Shuffle the training data
ind = randperm(n_train);
xTrain_copy = xTrain;
yTrain_copy = yTrain;
xTrain = xTrain(:, ind);
yTrain = yTrain(ind);
