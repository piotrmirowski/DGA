% KNN_TrainTest  Cross-validate, train and test K-Nearest Neighbors
%
% Syntax:
%   [measure_test, measure_train, yTest_pred, yTrain_pred, ...
%     net, n_hidden] = ...
%     KNN_TrainTest(xTrain, yTrain, xTest, yTest, paramsKNN)
% Inputs:
%   xTrain:       matrix of size <dim_x> x <n1> of input features (train)
%   yTrain:       vector of length <n1> of labels, train data
%   xTest:        matrix of size <dim_x> x <n2> of input features (test)
%   yTest:        vector of length <n2> of labels, train data
%   paramsKNN:    parameter struct with optional number of nearest neighbors
% Outputs:
%   measure_test:  accuracy on test data
%   measure_train: accuracy on train data
%   yTest_pred:    vector of length <n2> of predictions on test data
%   yTrain_pred:   vector of length <n1> of predictions on train data
%   net:           structure containing the trained nearest neighbors graph
%   n_hidden:      optimal cross-validated number of hidden units
%
% References:
%   T. Cover and P. Hart, "Nearest neighbor pattern classification",
%   IEEE Transactions on Information Theory, 1967.


% Copyright (C) 2010 Piotr Mirowski
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
% Version 1.0, New York, 24 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [measure_test, measure_train, yTest_pred, yTrain_pred, ...
  net, n_hidden] = ...
  KNN_TrainTest(xTrain, yTrain, xTest, yTest, paramsKNN)


% Parameters
% ----------

dim_x = size(xTrain, 1);
dim_y = size(yTrain, 1);

% Starting points for fine-grid hyperparameter optimization
if ((nargin == 5) && isfield(paramsKNN, 'n_hidden0'))
  n_hidden0 = paramsKNN.n_hidden0;
end


% Data preparation
% ----------------

% Data are assumed to be already scaled w.r.t. the learning task

% Format the data properly
n_train = length(yTrain);
yTrain = logical(yTrain);
if (size(yTrain, 1) == 1)
  yTrain = yTrain';
end
if (size(xTrain, 1) ~= n_train)
  xTrain = xTrain';
  if (size(xTrain, 1) ~= n_train)
    error('The size of xTrain does not match the size of yTrain');
  end
end
n_test = length(yTest);
yTest = logical(yTest);
if (size(yTest, 1) == 1)
  yTest = yTest';
end
if (size(xTest, 1) ~= n_test)
  xTest = xTest';
  if (size(xTest, 1) ~= n_test)
    error('The size of xTest does not match the size of yTest');
  end
end

% Shuffle the training data
[dummy, ind] = sort(rand(1, n_train));
xTrain_copy = xTrain;
yTrain_copy = yTrain;
xTrain = xTrain(ind, :);
yTrain = yTrain(ind);


% Hyperparameter selection (<n_hidden> nodes)
% --------------------------------------------------------------------

% Optimize for n_hidden on coarse grid to get n_hidden0
if exist('n_hidden0', 'var')
  fprintf(1, 'KNN: skip <n_hidden> coarse grid search...\n');
  cv_measure0 = 0;
  skip_grid = [];
else
  fprintf(1, 'KNN: optimizing <n_hidden> on coarse grid...\n');
  nHiddens = [1 3 10 30];
  [n_hidden0, cv_measure0] = GridSearch(nHiddens, xTrain, yTrain, []);
  skip_grid = [];
end
fprintf(1, 'KNN: n_hidden=%d\n', n_hidden0);

% Optimize for n_hidden on fine grid around n_hidden0
fprintf(1, 'KNN: optimization for n_hidden on the fine grid...\n');
n_hidden_min = min(nHiddens);
nHiddens = unique(max(n_hidden_min, round(n_hidden0 * 2.^[-2:2])));
[n_hidden, cv_measure] = GridSearch(nHiddens, xTrain, yTrain, skip_grid);

% Handle cases when the fine-grid cross-validation did not yield better
% results
if (cv_measure0 > cv_measure)
  n_hidden = n_hidden0;
end


% Train the SVM model
% -------------------
% Create a new KNN with <n_hidden> nodes
net = knn(dim_x, dim_y, n_hidden, xTrain, logical(yTrain));
% Retrieve the normal order
xTrain = xTrain_copy;
yTrain = yTrain_copy;
% Evaluate the KNN on training set
yTrain_pred = knnfwd(net, xTrain);
measure_train = Evaluate(yTrain_pred, yTrain);


% Test the SVM
% ------------

% Evaluate the KNN on cross-validation set
yTest_pred = knnfwd(net, xTest);
measure_test = Evaluate(yTest_pred, yTest);

% Print prediction results
fprintf(1, 'KNN: accuracy train %.2f%% and test %.2f%%\n', ...
  measure_train, measure_test);



% -------------------------------------------------------------------------
function [n_hidden, cv_measure] = ...
  GridSearch(nHiddens, xTrain, yTrain, skip)

dim_x = size(xTrain, 2);
dim_y = size(yTrain, 2);

% Define the cross-validation and learning sets
n_samples = size(yTrain, 1);
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
  xLearn{k} = xTrain(indLearn{k}, :);
  xXval{k} = xTrain(indXval{k}, :);
  yLearn{k} = yTrain(indLearn{k});
  yXval{k} = yTrain(indXval{k});
end


% Grid search for parameters n_hidden and lambda on a grid, using 5-fold
% cross-validation
measure = zeros(1, length(nHiddens));
for i = 1:length(nHiddens)
  n_hidden = nHiddens(i);
  if (~isempty(skip) && ~isempty(find((skip == i))))
    fprintf(1, 'KNN: Skipping n_hidden=%d...\n', n_hidden);
  else
    % 5-fold cross-validation
    fprintf(1, 'KNN: Cross-validating n_hidden=%d...\n', n_hidden);
    measureXval = zeros(1, 5);
    for k = 1:5
      % Create a new KNN with <n_hidden> nodes
      net = knn(dim_x, dim_y, n_hidden, xLearn{k}, yLearn{k});
      % Evaluate the KNN on cross-validation set
      yXval_Pred = knnfwd(net, xXval{k});
      measureXval(k) = Evaluate(yXval_Pred, yXval{k});
    end
    measure(i) = mean(measureXval);
    fprintf(1, '    measure=%g\n', measure(i));
  end
end

% Select the optimal hyperparameters
cv_measure = max(measure);
ind = find(measure == cv_measure);
n_hidden = nHiddens(ind);

% Take the median optimal hyperparameter (if ambiguity)
n = length(ind);
str = '';
if (n > 1)
  str = '[';
  for i = 1:n
    str = sprintf('%s(%g) ', str, n_hidden(i));
  end
  str = sprintf('%s] ', str);
  n_hidden = n_hidden(ceil(n/2));
end

fprintf(1, 'Best n_hidden=%g, %s(accuracy %g%%)\n', ...
  n_hidden, str, cv_measure);


% -------------------------------------------------------------------------
function measure = Evaluate(yPred, y)

measure = sum((yPred > 0.5) == y) / length(y) * 100;
