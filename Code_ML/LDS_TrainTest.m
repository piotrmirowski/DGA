% LDS_TrainTest  Cross-validate, train and test Low Dimensional Scaling
%
% Syntax:
%   [measure_test, yTest_pred, C, rho] = ...
%     LDS_TrainTest(xTrain, yTrain, xTest, yTest)
% Inputs:
%   xTrain:       matrix of size <dim_x> x <n1> of input features (train)
%   yTrain:       vector of length <n1> of labels, train data
%   xTest:        matrix of size <dim_x> x <n2> of input features (test)
%   yTest:        vector of length <n2> of labels, train data
% Outputs:
%   measure_test:  mean square error or accuracy on test data
%   yTest_pred:    vector of length <n2> of predictions on test data
%   C:             optimal cross-validated regularization parameter C
%   rho:           optimal cross-validated LDS parameter rho
%
% This function requires the LDS library, available at:
%   http://olivier.chapelle.cc/lds/
%
% References:
%   O. Chapelle and A. Zien,
%   "Semi-supervised classification by low density separation",
%   in Proceedings of AISTATS?05, 2005.

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

function [measure_test, yTest_pred, C, rho] = ...
  LDS_TrainTest(xTrain, yTrain, xTest, yTest)


% Data preparation
% ----------------

% Data are assumed to be already scaled w.r.t. the learning task
% Format the data properly, and shuffle them
[xTrain, yTrain, xTest, yTest] = ...
  ML_CheckShuffleData(xTrain, yTrain, xTest, yTest);

if (islogical(yTrain) && islogical(yTest))
  yTrain = sign(double(yTrain) - 0.5);
  yTest = sign(double(yTest) - 0.5);
end

% Hyperparameter selection (regularization <lambda>, <n_hidden> nodes)
% --------------------------------------------------------------------

% Optimize for n_hidden and lambda on a grid
fprintf(1, 'LDS: optimization <h> and <gamma> on a grid...\n');
Cs = 10.^[-1:2];
rhos = [0 1 2 4 8 inf];
[C, rho] = GridSearch(Cs, rhos, xTrain, yTrain, []);


% Train and test the weighted kernel model
% ----------------------------------------

% Parameters of LDS
opt.C = C;
opt.delta = 0.1;
opt.sigma = 1;

% Create a new LDS, make the prediction on the test set
yTest_pred = lds(xTrain, xTest, yTrain', rho, opt);
measure_test = ML_MeasurePerformance(yTest, sign(yTest_pred'), 0);

% Print prediction results
fprintf(1, 'LDS: accuracy test %.4f%%\n', measure_test);


% -------------------------------------------------------------------------
function [C, rho, cv_measure] = ...
  GridSearch(Cs, rhos, xTrain, yTrain, skip)

% Define the cross-validation and learning sets
[xLearn, yLearn, xXval, yXval] = ML_SplitLearnXval(xTrain, yTrain);

% Grid search for parameters n_hidden and lambda on a grid, using 5-fold
% cross-validation
measure = zeros(length(Cs), length(rhos));
for i = 1:length(Cs)
  C = Cs(i);

  % Parameters of LDS
  opt.C = C;
  opt.delta = 0.1;
  opt.sigma = 1;
  opt.verb = 0;
  opt.maxiter = 10;
  
  for j = 1:length(rhos)
    rho = rhos(j);
    if (~isempty(skip) && ...
        ~isempty(find((skip(:, 1) == i) & (skip(:, 2) == j), 1)))
      fprintf(1, 'LDS: Skipping C=%g, rho=%g...\n', C, rho);
    else
      % 5-fold cross-validation
      fprintf(1, 'LDS: Cross-validating C=%g, rho=%g...\n', C, rho);
      measureXval = zeros(1, 5);
      k = 1;
      cont = 1;
      while cont
        % Perform weighted kernel regression with bandwidth <h> and evaluate
        % the performance on the cross-validation set
        yXval_Pred = ...
          lds(xLearn{k}, xXval{k}, yLearn{k}', rho, opt);
        yXval_Pred = sign(yXval_Pred');
        measureXval(k) = ML_MeasurePerformance(yXval{k}, yXval_Pred, 0);
        if ((k >= 5) || isnan(measureXval(k)))
          cont = 0;
        else
          k = k + 1;
        end
      end
      measure(i, j) = mean(measureXval);
      fprintf(1, '    measure=%g\n', measure(i, j));
    end
  end
end

% Select the optimal hyperparameters
cv_measure = max(measure(:));
ind = find(measure == cv_measure);
[rhoGrid, CGrid] = meshgrid(rhos, Cs);
C = CGrid(ind);
rho = rhoGrid(ind);

% Take the median optimal hyperparameter (if ambiguity)
n = length(ind);
str = '';
if (n > 1)
  str = '[';
  for i = 1:n
    str = sprintf('%s(%g,%g) ', str, C(i), rho(i));
  end
  str = sprintf('%s] ', str);
  C = C(ceil(n/2));
  rho = rho(ceil(n/2));
end

fprintf(1, 'Best C=%g, rho=%g, %s(%.4f%%)\n', C, rho, str, cv_measure);
