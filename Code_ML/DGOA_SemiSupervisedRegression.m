% DGOA_LinearRegression  Perform linear regression with cross-validation
%
% Syntax:
%   [w, b, nmse, nmse_xval, y_reg] = DGOA_LinearRegression(x, y)
% Inputs:
%   x: matrix of inputs of size <N> x <T>
%   y: vector of outputs of size 1 x <T>
% Outputs:
%   w:         vector of size <N> x 1 of weights
%   b:         bias
%   nmse:      normalized mean square error
%   nmse_xval: normalized mean cross-validation square error
%   y_reg:     regressed outputs
%   type:      string of char indicating the type of regression:
%              'linear' or 'lasso'

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
% Version 1.0, New York, 08 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [w, b, nmse, nmse_xval, y_pos_reg, y_neg_reg] = ...
  DGOA_SemiSupervisedRegression(x_pos, y_pos, x_neg, y_neg, d, r)

% Get the number of samples size and check the size of the inputs <x>
n_var = size(x_pos, 1);
n_pos = size(x_pos, 2);
n_neg = size(x_neg, 2);
if (length(y_pos) ~= n_pos)
  error('<y_pos> needs to have length %d', n_pos);
end
if (length(y_neg) ~= n_neg)
  error('<y_neg> needs to have length %d', n_neg);
end

% Create feature vectors
if (d == 1)
  fx_pos = x_pos;
  fx_neg = x_neg;
elseif (d == 2)
  fx_pos = x_pos.^2;
  fx_neg = x_neg.^2;
  for i = 1:n_var
    for j = (i+1):n_var
      fx_pos = [fx_pos; sqrt(2) * x_pos(i, :) .* x_pos(j, :)];
      fx_neg = [fx_neg; sqrt(2) * x_neg(i, :) .* x_neg(j, :)];
    end
  end
  if (r > 0)
    fx_pos = [fx_pos; r^2 * ones(1, n_pos); sqrt(2) * r * x_pos];
    fx_neg = [fx_neg; r^2 * ones(1, n_neg); sqrt(2) * r * x_neg];
  end
else
  error('Not implemented.');
end

% Perform a grid search on the regularization
nmse_best = inf;
for C = 2.^[-15:1]
  for lambda =  2.^[-9:5]
    % Cross-validate this value of <lambda>
    nmse_xval = ...
      mean(CrossValidate(fx_pos, y_pos, fx_neg, y_neg, C, lambda));

    % Results on the cross-validations
    fprintf(1, 'C=%g, lambda=%g, x-val NMSE=%.4f', C, lambda, nmse_xval);
    if (nmse_xval < nmse_best)
      nmse_best = nmse_xval;
      lambda_best = lambda;
      C_best = C;
      fprintf(1, ' ***');
    end
    fprintf(1, '\n');
  end
end
fprintf(1, 'Best hyperparameters: C = %g (2^%g), lambda = %g (2^%g)\n', ...
  C_best, log2(C_best), lambda_best, log2(lambda_best));

% Final result
[nmse_xval, nmse, w, b, y_pos_reg, y_neg_reg] = ...
  CrossValidate(fx_pos, y_pos, fx_neg, y_neg, C_best, lambda_best);

% Results on the cross-validations
fprintf(1, 'Average NMSE over 10-fold x-val: %g\n', mean(nmse_xval));

% Results on the full training data
fprintf(1, 'NMSE over trainning data: %g\n', nmse);
fprintf(1, 'Value of <w> on the training set:\n');
disp(w');
fprintf(1, 'Value of <b> on the training set:\n');
disp(b);


% -------------------------------------------------------------------------
function [nmse_xval, nmse, w, b, y_pos_reg, y_neg_reg] = ...
  CrossValidate(x_pos, y_pos, x_neg, y_neg, C, lambda)

% Create random selection of positive samples
n_pos = length(y_pos);
[dummy, indRandom] = sort(rand(1, n_pos));
x_pos_rand = x_pos(:, indRandom);
y_pos_rand = y_pos(indRandom);

% 10-fold cross-validation
n_pos_test = ceil(n_pos / 10);
n_var = size(x_pos_rand, 1);
k = 0;
w_xval = zeros(n_var, 10);
b_xval = zeros(1, 10);
nmse_xval = zeros(1, 10);
for i = 1:n_pos_test:n_pos
  k = k + 1;

  % Select cross-validation samples
  indKxVal = i:(i+n_pos_test-1);
  indKxVal = unique(min(indKxVal, n_pos));
  % Select training input data
  x_xval = x_pos_rand(:, indKxVal);
  % Select output training data
  y_xval = y_pos_rand(indKxVal);
  % Select training samples
  indKtrain = setdiff(1:n_pos, indKxVal);
  % Select training input data
  x_pos_k = x_pos_rand(:, indKtrain);
  % Select output training data
  y_pos_k = y_pos_rand(indKtrain);

  % Do the semi-supervised linear regression on the training set
  [w_xval(:, k), b_xval(k)] = ...
    SemiSupervisedLinearRegression(x_pos_k, y_pos_k, x_neg, y_neg, ...
    C, lambda);
  
  % Compute the estimate and the associated cross-validation NMSE
  nmse_xval(k) = RegressNMSE(w_xval(:, k), b_xval(k), x_xval, y_xval);
  fprintf(1, '.');
end
fprintf(1, '\n');

if (nargout > 1)
  % Do the semi-supervised linear regression per se
  [w, b, nmse, y_pos_reg, y_neg_reg] = ...
    SemiSupervisedLinearRegression(x_pos, y_pos, x_neg, y_neg, ...
    C, lambda);
end


% -------------------------------------------------------------------------
function [w, b, nmse_pos, y_pos_reg, y_neg_reg] = ...
  SemiSupervisedLinearRegression(x_pos, y_pos, x_neg, y_neg, ...
  C, lambda)

n_var = size(x_pos, 1);
n_pos = size(x_pos, 2);
n_neg = size(x_neg, 2);

% Transductive regression (cf. Cortes & Mohri, 2006)
x_pos = [ones(1, n_pos); x_pos];
x_neg = [ones(1, n_neg); x_neg];
w = inv(lambda * eye(n_var+1) + x_pos * x_pos' + C * x_neg * x_neg') * ...
  (x_pos * y_pos' + C * x_neg * y_neg');

% Separate bias from weights
b = w(1);
w = w(2:end);
x_pos = x_pos(2:end, :);
x_neg = x_neg(2:end, :);

% Compute the estimate of y and the associated NMSE
[nmse_pos, y_pos_reg] = RegressNMSE(w, b, x_pos, y_pos);
[nmse_neg, y_neg_reg] = RegressNMSE(w, b, x_neg, y_neg);


% -------------------------------------------------------------------------
function [nmse, y_reg] = RegressNMSE(w, b, x, y)

y_reg = w' * x + b;
nmse = mean((y - y_reg).^2) / mean(y.^2);
