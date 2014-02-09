% NN_TrainTest  Cross-validate, train and test Neural Networks
%
% Syntax:
%   [measure_test, measure_train, yTest_pred, yTrain_pred, ...
%     net, n_hidden, lambda] = ...
%     NN_TrainTest(xTrain, yTrain, xTest, yTest, paramsNN)
% Inputs:
%   xTrain:       matrix of size <dim_x> x <n1> of input features (train)
%   yTrain:       vector of length <n1> of labels, train data
%   xTest:        matrix of size <dim_x> x <n2> of input features (test)
%   yTest:        vector of length <n2> of labels, train data
%   paramsNN:     parameter struct. Default values are:
%                 paramsNN.output = 'logistic' (for classification)
%                 Other possible values are 'linear' (for regression)
%                 Look at function NN_Params_Init for more explanations
% Outputs:
%   measure_test:  mean square error or accuracy on test data
%   measure_train: mean square error or accuracy on train data
%   yTest_pred:    vector of length <n2> of predictions on test data
%   yTrain_pred:   vector of length <n1> of predictions on train data
%   net:           structure containing the trained NN
%   n_hidden:      optimal cross-validated number of hidden units
%   lambda:        optimal cross-validated regularization parameter
%
% References:
%   D. E. Rumelhart, G. E. Hinton, and R. J. Williams, 
%   Learning internal representations by error propagation.	
%   Cambridge, MA, USA: MIT Press, 1986, pp. 318?362.
%   Y. LeCun, L. Bottou, G. Orr, and K. Muller, "Efficient backprop",
%   in Lecture Notes in Computer Science. Berlin/Heidelberg: Springer, 1998
%   L. Bottou, "Stochastic learning",
%   in Advanced Lectures on Machine Learning, Springer-Verlag, 2004.

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
  net, n_hidden, lambda] = ...
  NN_TrainTest(xTrain, yTrain, xTest, yTest, paramsNN)


% Parameters
% ----------

% Retrieve parameters or use ones by default
if (nargin < 5)
  % Logistic regression for classification
  paramsNN.output = 'logistic';
end
output = paramsNN.output;
is_linear = isequal(output, 'linear');

% Starting points for fine-grid hyperparameter optimization
if isfield(paramsNN, 'n_hidden0')
  n_hidden0 = paramsNN.n_hidden0;
end
if isfield(paramsNN, 'lambda0')
  lambda0 = paramsNN.lambda0;
end


% Data preparation
% ----------------

% Data are assumed to be already scaled w.r.t. the learning task
% Format the data properly, and shuffle them
[xTrain, yTrain, xTest, yTest, xTrain_copy, yTrain_copy] = ...
  ML_CheckShuffleData(xTrain, yTrain, xTest, yTest);


% Hyperparameter selection (regularization <lambda>, <n_hidden> nodes)
% --------------------------------------------------------------------

% Optimize for n_hidden and lambda on coarse grid to get n_hidden0 and lambda0
if (exist('n_hidden0', 'var') && exist('lambda0', 'var'))
  fprintf(1, 'MLP: skip <n_hidden> and <lambda> coarse grid search...\n');
  cv_measure0 = 0;
  skip_grid = [];
else
  fprintf(1, 'MLP: optimizing <n_hidden> and <lambda> on coarse grid...\n');
  nHiddens = [1 3 10 30];
  lambdas = 10.^[-5 -4 -3 -2 -1 0];
  [n_hidden0, lambda0, cv_measure0] = ...
    GridSearch(nHiddens, lambdas, xTrain, yTrain, [], paramsNN);
  skip_grid = [];
end
fprintf(1, 'MLP: n_hidden=%d, lambda0=%g\n', n_hidden0, lambda0);

% Optimize for n_hidden and lambda on fine grid around n_hidden0 and lambda0
fprintf(1, 'MLP: optimization for n_hidden and lambda on the fine grid...\n');
lambdas = lambda0 * 2.^[-2:2:2];
n_hidden_min = 1;
nHiddens = unique(min(max(n_hidden_min, round(n_hidden0*2.^[-2:2])), 30));
[n_hidden, lambda, cv_measure] = ...
  GridSearch(nHiddens, lambdas, xTrain, yTrain, skip_grid, paramsNN);

% Handle cases when the fine-grid cross-validation did not yield better
% results
if isequal(output, 'linear')
  if (cv_measure0 < cv_measure)
    n_hidden = n_hidden0;
    lambda = lambda0;
  end
else
  if (cv_measure0 > cv_measure)
    n_hidden = n_hidden0;
    lambda = lambda0;
  end
end


% Train the SVM model
% -------------------

% Create default parameters for the MLP, set <n_hidden>, <lambda>
params = NN_Params_Init();
params.lambda = lambda;
params.dim_z = n_hidden;
params.perc_xval = 0;
if isequal(output, 'linear')
  params.energy = 'gaussian';
  params.output = 'linear';
end
% Create a new MLP with <n_hidden> nodes and train it with x-val
net = NN_Trainer(xTrain, yTrain, params);

% Retrieve the normal order
xTrain = xTrain_copy;
yTrain = yTrain_copy;

% Use the MLP to make predictions on the training set
yTrain_pred = NN_Predict(xTrain, net);
measure_train = ML_MeasurePerformance(yTrain, yTrain_pred, is_linear);


% Test the SVM
% ------------

% Evaluate the MLP on test set
yTest_pred = NN_Predict(xTest, net);
measure_test = ML_MeasurePerformance(yTest, yTest_pred, is_linear);

% Print prediction results
if isequal(output, 'linear')
  fprintf(1, 'MLP: train R2 %g and test R2 %g\n', ...
    measure_train, measure_test);
else
  fprintf(1, 'MLP: accuracy train %.2f%% and test %.2f%%\n', ...
    measure_train, measure_test);
end



% -------------------------------------------------------------------------
function [n_hidden, lambda, cv_measure] = ...
  GridSearch(nHiddens, lambdas, xTrain, yTrain, skip, paramsNN)

% Retrieve parameters
output = paramsNN.output;

% Grid search for parameters n_hidden and lambda on a grid, using 5-fold
% cross-validation
measure = zeros(length(nHiddens), length(lambdas));
for i = 1:length(nHiddens)
  n_hidden = nHiddens(i);
  for j = 1:length(lambdas)
    lambda = lambdas(j);
    if (~isempty(skip) && ...
        ~isempty(find((skip(:, 1) == i) & (skip(:, 2) == j), 1)))
      fprintf(1, 'MLP: Skipping n_hidden=%g, lambda=%g...\n', ...
        n_hidden, lambda);
    else
      % 5-fold cross-validation
      fprintf(1, 'MLP: Cross-validating n_hidden=%g, lambda=%g...\n', ...
        n_hidden, lambda);
      measureXval = zeros(1, 5);
      for k = 1:5
        if 1
          % Create default parameters for the MLP, set <n_hidden>, <lambda>
          params = NN_Params_Init();
          params.lambda = lambda;
          params.dim_z = n_hidden;
          params.perc_xval = 0.2;
          params.n_epochs = 200;
          if isequal(output, 'linear')
            params.energy = 'gaussian';
            params.output = 'linear';
          end
          % Create a new MLP with <n_hidden> nodes and train it with x-val
          % Evaluate the MLP on cross-validation set
          [net, measureXval(k)] = NN_Trainer(xTrain, yTrain, params);
        else
          % Create a new MLP with <n_hidden> nodes
          net = mlp(dim_x, n_hidden, dim_y, output, lambda);
          % Train the MLP
          net = netopt(net, options, xLearn{k}, yLearn{k}, algorithm);
          % Evaluate the MLP on cross-validation set
          yXval_Pred = mlpfwd(net, xXval{k});
          measureXval(k) = ...
            ML_MeasurePerformance(yXval{k}, yXval_Pred, ...
            isequal(output, 'linear'));
        end
      end
      measure(i, j) = mean(measureXval);
      fprintf(1, '    measure=%g\n', measure(i, j));
    end
  end
end

% Select the optimal hyperparameters
if isequal(output, 'linear')
  % In the case of regression, we choose the lowest MSE
  cv_measure = min(measure(:));
else
  % In the case of classification, we choose the highest accuracy
  cv_measure = max(measure(:));
end
ind = find(measure == cv_measure);
[lambdaGrid, nHiddenGrid] = meshgrid(lambdas, nHiddens);
n_hidden = nHiddenGrid(ind);
lambda = lambdaGrid(ind);

% Take the median optimal hyperparameter (if ambiguity)
n = length(ind);
str = '';
if (n > 1)
  str = '[';
  for i = 1:n
    str = sprintf('%s(%g,%g) ', str, n_hidden(i), lambda(i));
  end
  str = sprintf('%s] ', str);
  n_hidden = n_hidden(ceil(n/2));
  lambda = lambda(ceil(n/2));
end

if isequal(output, 'linear')
  fprintf(1, 'Best n_hidden=%g, best lambda=%g %s(R2 %g)\n', ...
    n_hidden, lambda, str, cv_measure);
else
  fprintf(1, 'Best n_hidden=%g, best lambda=%g %s(accuracy %g%%)\n', ...
    n_hidden, lambda, str, cv_measure);
end
