% SVM_TrainTest  Cross-validate, train and test an SVM
%
% Syntax:
%   [measure_test, measure_train, y_test_pred, y_train_pred, ...
%     modelSVM, C, gamma, y_test_prob, y_train_prob] = ...
%     SVM_TrainTest(x_train, y_train, x_test, y_test, fileNameSVM, paramSVM)
% Inputs:
%   x_train:       matrix of size <n1> x <dim_x> of input features (train)
%   y_train:       vector of length <n1> of labels, train data
%   x_test:        matrix of size <n2> x <dim_x> of input features (test)
%   y_test:        vector of length <n2> of labels, train data
%   fileNameSVM:   filename where the temporary results will be saved
%   paramSVM:      parameter struct for LibSVM
%                  Default values are:
%                  paramSVM.epsilon = 0.01; % tolerated regression error
%                  paramSVM.type = 3;       % epsilon-SVR regression
%                  paramSVM.kernel = 2;     % Gaussian kernel
%                  Other values can be:
%                  paramSVM.kernel = 0      % for linear kernel
%                  paramSVM.kernel = 1      % for quadratic kernel
%                  paramSVM.type = 0;       % classification
% Outputs:
%   measure_test:  mean square error or accuracy on test data
%   measure_train: mean square error or accuracy on train data
%   y_test_pred:   vector of length <n2> of predictions on test data
%   y_train_pred:  vector of length <n1> of predictions on train data
%   modelSVM:      model returned by LibSVM's training function
%   C:             optimal cross-validated regularization parameter C
%   gamma:         optimal cross-validated Gaussian kernel parameter gamma
%   y_test_pred:   vector of length <n2> of proba. estimates on test data
%   y_train_pred:  vector of length <n1> of proba. estim. on train data
%
% This interface relies on the LibSVM library:
%   Chih-Chung Chang and Chih-Jen Lin, 2001
%   "LIBSVM: a library for support vector machines"
%   available at: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
%
% References:
%   C.Cortes & V.Vapnik, "Support vector networks",
%   Machine Learning, 1995.
%   A. J. Smola & B. Schlkopf, "A tutorial on support vector regression",
%   Statistics and Computing, vol. 14, pp. 199?222, 2004

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

function [measure_test, measure_train, y_test_pred, y_train_pred, ...
  modelSVM, C, gamma, y_test_prob, y_train_prob] = ...
  SVM_TrainTest(x_train, y_train, x_test, y_test, fileNameSVM, paramSVM)


% Parameters
% ----------

% Retrieve parameters or use ones by default
if (nargin < 6)
  paramSVM.epsilon = 0.01;
  paramSVM.type = 3;   % epsilon-SVR
  paramSVM.kernel = 2; % Gaussian
end
typeSVM = paramSVM.type;
kernelSVM = paramSVM.kernel;
try
  epsilonSVR = paramSVM.epsilon;
catch
  epsilonSVR = 0;
end

% Error checking
if ~ismember(typeSVM, [0 2 3])
  fprintf(1, 'Error: typeSVM = %g\n', typeSVM);
  error('Only C-SVM (0), one-class SVM (2) or epsilon-SVR (3) supported');
end
if ~ismember(kernelSVM, [0 1 2])
  fprintf(1, 'Error: kernelSVM = %g\n', kernelSVM);
  error('Only linear (0), quadratic (1) or gaussian (2) SVM supported');
end

% Starting points for fine-grid hyperparameter optimization
if isfield(paramSVM, 'C0')
  C0 = paramSVM.C0;
end
if isfield(paramSVM, 'gamma0')
  gamma0 = paramSVM.gamma0;
end


% Data preparation
% ----------------

% Data are assumed to be already scaled w.r.t. the learning task

% Format the data properly
nTrain = length(y_train);
if (size(y_train, 1) == 1)
  y_train = y_train';
end
if (size(x_train, 1) ~= nTrain)
  x_train = x_train';
  if (size(x_train, 1) ~= nTrain)
    error('The size of x_train does not match the size of y_train');
  end
end
nTest = length(y_test);
if (size(y_test, 1) == 1)
  y_test = y_test';
end
if (size(x_test, 1) ~= nTest)
  x_test = x_test';
  if (size(x_test, 1) ~= nTest)
    error('The size of x_test does not match the size of y_test');
  end
end

% Shuffle the training data
[dummy, ind] = sort(rand(1, nTrain));
x_train_copy = x_train;
y_train_copy = y_train;
x_train = x_train(ind, :);
y_train = y_train(ind);


% Hyperparameter selection (regularization C, gaussian width gamma)
% -----------------------------------------------------------------

% Optimize for C and gamma on coarse grid to get C0 and gamma0
if (exist('C0', 'var') && exist('gamma0', 'var'))
  fprintf(1, 'SVM: skipping C and gamma search on the coarse grid...\n');
  cvMeasure0 = 0;
  skipGrid = [];
else
  fprintf(1, 'SVM: optimizing C and gamma on the coarse grid...\n');
  Cs = 2.^[-1:2:7];
  if (kernelSVM == 2)
    gammas = 2.^[-9:2:3];
  else
    gammas = 1;
  end
  [C0, gamma0, cvMeasure0] = ...
    GridSearch(Cs, gammas, x_train, y_train, [], paramSVM);
  skipGrid = [1 1; 1 5; 1 9; 5 1; 5 5; 5 9; 9 1; 9 5; 9 9];
end
fprintf(1, 'SVM: C0=%g, gamma0=%g\n', C0, gamma0);

% Optimize for C and gamma on fine grid around C0 and gamma0
fprintf(1, 'SVM: optimization for C and gamma on the fine grid...\n');
Cs = C0 * 2.^[-2:0.5:2];
if (kernelSVM == 2)
  gammas = gamma0 * 2.^[-2:0.5:2];
else
  gammas = 1;
end
[C, gamma, cvMeasure] = ...
  GridSearch(Cs, gammas, x_train, y_train, skipGrid, paramSVM);

% Handle cases when the fine-grid cross-validation did not yield better
% results
if (typeSVM == 3)
  if (cvMeasure0 < cvMeasure)
    C = C0;
    gamma = gamma0;
    cvMeasure = cvMeasure0;
  end
else
  if (cvMeasure0 < cvMeasure)
    C = C0;
    gamma = gamma0;
    cvMeasure = cvMeasure0;
  end
end


% Train the SVM model
% -------------------
paramStr = sprintf('-s %d -t %d -g %g -c %g -b 1', ...
  typeSVM, kernelSVM, gamma, C);
if (epsilonSVR > 0)
  paramStr = sprintf('%s -p %g', paramStr, epsilonSVR);
end
if (kernelSVM == 1)
  paramStr = sprintf('%s -d 2 -r 1 -g 1.4142', paramStr);
end
modelSVM = svmtrain(y_train, x_train, paramStr);
totalSV = modelSVM.totalSV;
fprintf(1, '%d SV found\n', totalSV);
if (typeSVM ~= 3)
  nSV = modelSVM.nSV;
  for k = 1:modelSVM.nr_class
    fprintf(1, 'Class %d: %d SV\n', k, nSV(k));
  end
end


% Test the SVM
% ------------
x_train = x_train_copy;
y_train = y_train_copy;

% Try with probability estimates
[y_train_pred, measure_train, y_train_prob] = ...
  svmpredict(y_train, x_train, modelSVM, '-b 1');
[y_test_pred, measure_test, y_test_prob] = ...
  svmpredict(y_test, x_test, modelSVM, '-b 1');

% If needed, try without probability estimates
if isempty(y_train_pred)
  [y_train_pred, measure_train] = ...
    svmpredict(y_train, x_train, modelSVM, '-b 0');
  [y_test_pred, measure_test] = ...
    svmpredict(y_test, x_test, modelSVM, '-b 0');
end

% Print prediction results
if (typeSVM == 3)
  fprintf(1, 'SVM: train MSE %g, R2 %g and test MSE %g, R2 %g\n', ...
    measure_train(2), measure_train(3), measure_test(2), measure_test(3));
  measure_train = measure_train(3);
  measure_test = measure_test(3);
else
  measure_train = measure_train(1);
  measure_test = measure_test(1);
  fprintf(1, 'SVM: accuracy train %.2f%% and test %.2f%%\n', ...
    measure_train, measure_test);
end


% Save the results
% ----------------
save(fileNameSVM, 'modelSVM', 'C', 'gamma', ...
  'y_test_pred', 'y_test', 'y_train_pred', 'y_train', ...
  'measure_train', 'measure_test');


% -------------------------------------------------------------------------
function [C, gamma, cvMeasure] = ...
  GridSearch(Cs, gammas, x_train, y_train, skip, paramSVM)

% Retrieve parameters
try
  epsilonSVR = paramSVM.epsilon;
catch
  epsilonSVR = 0;
end
typeSVM = paramSVM.type;
kernelSVM = paramSVM.kernel;

% Grid search for parameters C and gamma on a grid, using 5-fold
% cross-validation
measure = zeros(length(Cs), length(gammas));
for i = 1:length(Cs)
  C = Cs(i);
  for j = 1:length(gammas)
    gamma = gammas(j);
    if (~isempty(skip) && ...
        ~isempty(find((skip(:, 1) == i) & (skip(:, 2) == j), 1)))
      fprintf(1, 'SVM: Skipping C=%g, gamma=%g...\n', C, gamma);
    else
      fprintf(1, 'SVM: Cross-validating C=%g, gamma=%g...\n', C, gamma);
      paramStr = sprintf('-s %d -t %d -g %g -c %g -v 5', ...
        typeSVM, kernelSVM, gamma, C);
      if (epsilonSVR > 0)
        paramStr = sprintf('%s -p %g', paramStr, epsilonSVR);
      end
      if (kernelSVM == 1)
        paramStr = sprintf('%s -d 2 -r 1 -g 1.4142', paramStr);
      end
      measure(i, j) = svmtrain(y_train, x_train, paramStr);
    end
  end
end

% Select the optimal hyperparameters
if (typeSVM == 3)
  % In the case of regression, we choose the lowest MSE
  cvMeasure = min(measure(:));
else
  % In the case of classification, we choose the highest accuracy
  cvMeasure = max(measure(:));
end
ind = find(measure == cvMeasure);
[gammaGrid, CGrid] = meshgrid(gammas, Cs);
C = CGrid(ind);
gamma = gammaGrid(ind);

% Take the median optimal hyperparameter (if ambiguity)
n = length(ind);
str = '';
if (n > 1)
  str = '[';
  for i = 1:n
    str = sprintf('%s(%g,%g) ', str, C(i), gamma(i));
  end
  str = sprintf('%s] ', str);
  C = C(ceil(n/2));
  gamma = gamma(ceil(n/2));
end

if (typeSVM == 3)
  fprintf(1, 'Best C=%g, best gamma=%g %s(MSE %g)\n', ...
    C, gamma, str, cvMeasure);
else
  fprintf(1, 'Best C=%g, best gamma=%g %s(accuracy %g)\n', ...
    C, gamma, str, cvMeasure);
end
