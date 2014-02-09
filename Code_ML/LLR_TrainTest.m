% LLR_TrainTest  Cross-validate, train and test Local Linear Regression
%
% Syntax:
%   [measure_test, yTest_pred, h] = ...
%     LLR_TrainTest(xTrain, yTrain, xTest, yTest, h_0)
% Inputs:
%   xTrain:       matrix of size <dim_x> x <n1> of input features (train)
%   yTrain:       vector of length <n1> of labels, train data
%   xTest:        matrix of size <dim_x> x <n2> of input features (test)
%   yTest:        vector of length <n2> of labels, train data
%   h_0:          optional initial bandwidth parameter
% Outputs:
%   measure_test: mean square error or accuracy on test data
%   yTest_pred:   vector of length <n2> of predictions on test data
%   h:            optimal cross-validated bandwidth parameter
%
% Reference:
%   C.J. Stone, Consistent Nonparametric Regression,
%   The Annals of Statistics, Vol. 5, No. 4. (Jul., 1977), pp. 595-620.
%   T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical
%   Learning, Chapter 6, Springer, 2001

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

function [measure_test, yTest_pred, h] = ...
  LLR_TrainTest(xTrain, yTrain, xTest, yTest, h_0)


% Data preparation
% ----------------

% Data are assumed to be already scaled w.r.t. the learning task
% Format the data properly, and shuffle them
[xTrain, yTrain, xTest, yTest] = ...
  ML_CheckShuffleData(xTrain, yTrain, xTest, yTest);


% Hyperparameter selection (regularization <lambda>, <n_hidden> nodes)
% --------------------------------------------------------------------

% Optimize for n_hidden and lambda on coarse grid to get h_0 and lambda0
if exist('h_0', 'var')
  fprintf(1, 'LLR: skip bandwidth <h> coarse grid search...\n');
  cv_measure0 = 0;
  skip_grid = [];
else
  fprintf(1, 'LLR: optimizing bandwidth <h> on coarse grid...\n');
  bandwidths = 2.^[-5:2:3];
  [h_0, cv_measure0] = GridSearch(bandwidths, xTrain, yTrain, []);
  skip_grid = [];
end
fprintf(1, 'LLR: bandwidth h=%d\n', h_0);

% Optimize for n_hidden and lambda on fine grid around h_0 and lambda0
fprintf(1, 'LLR: optimization for <h> on the fine grid...\n');
bandwidths = h_0 * 2.^[-2:0.5:2];
[h, cv_measure] = GridSearch(bandwidths, xTrain, yTrain, skip_grid);

% Handle cases when the fine-grid cross-validation did not yield better
% results
if (cv_measure0 > cv_measure)
  h = h_0;
end


% Train and test the weighted kernel model
% ----------------------------------------

% Create a new LLR with bandwidth <h>, make the prediction on the test set
yTest_pred = LLR_Predict(xTrain, yTrain, xTest, h);
measure_test = ML_MeasurePerformance(yTest, yTest_pred, 1);

% Print prediction results
fprintf(1, 'LLR: accuracy test %.4f\n', measure_test);


% -------------------------------------------------------------------------
function [h, cv_measure] = GridSearch(bandwidths, xTrain, yTrain, skip)

% Define the cross-validation and learning sets
[xLearn, yLearn, xXval, yXval] = ML_SplitLearnXval(xTrain, yTrain);

% Grid search for parameters n_hidden and lambda on a grid, using 5-fold
% cross-validation
measure = zeros(1, length(bandwidths));
for i = 1:length(bandwidths)
  h = bandwidths(i);
  if (~isempty(skip) && ~isempty(find((skip == i), 1)))
    fprintf(1, 'LLR: Skipping bandwidth=%g...\n', h);
  else
    % 5-fold cross-validation
    fprintf(1, 'LLR: Cross-validating bandwidth=%g...\n', h);
    measureXval = zeros(1, 5);
    for k = 1:5
      % Perform weighted kernel regression with bandwidth <h> and evaluate
      % the performance on the cross-validation set
      yXval_Pred = LLR_Predict(xLearn{k}, yLearn{k}, xXval{k}, h);
      measureXval(k) = ML_MeasurePerformance(yXval{k}, yXval_Pred, 1);
    end
    measure(i) = mean(measureXval);
    fprintf(1, '    measure=%g\n', measure(i));
  end
end

% Select the optimal hyperparameters
cv_measure = max(measure);
ind = find(measure == cv_measure);
h = bandwidths(ind);

% Take the median optimal hyperparameter (if ambiguity)
n = length(ind);
str = '';
if (n > 1)
  str = '[';
  for i = 1:n
    str = sprintf('%s(%g) ', str, h(i));
  end
  str = sprintf('%s] ', str);
  h = h(ceil(n/2));
end

fprintf(1, 'Best bandwidth=%g, %s(R2 %.4f)\n', h, str, cv_measure);
