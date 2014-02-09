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

function [w, b, nmse, nmse_xval, y_reg] = DGOA_LinearRegression(x, y, type)

% Get the number of samples size and check the size of the inputs <x>
n_samples = length(y);
if (size(x, 2) ~= n_samples)
  error('<x> needs to have %d columns', n_samples);
end

% By default, do the linear regression
if (nargin < 3)
  type = 'linear';
end

% Create random selection of samples
[dummy, indRandom] = sort(rand(1, n_samples));
x_rand = x(:, indRandom);
y_rand = y(indRandom);

% 10-fold cross-validation
n_samplesTest = ceil(n_samples / 10);
n_var = size(x_rand, 1);
k = 0;
w_xval = zeros(n_var, 10);
b_xval = zeros(1, 10);
nmse_xval = zeros(1, 10);
for i = 1:n_samplesTest:n_samples
  k = k + 1;

  % Select cross-validation samples
  indKxVal = i:(i+n_samplesTest-1);
  indKxVal = unique(min(indKxVal, n_samples));
  % Select cross-validation input data
  x_xval = x_rand(:, indKxVal);
  % Select output cross-validation data
  y_xval = y_rand(indKxVal);
  % Select training samples
  indKtrain = setdiff(1:n_samples, indKxVal);
  % Select training input data
  x_k = x_rand(:, indKtrain);
  % Select output training data
  y_k = y_rand(indKtrain);

  % Do the linear regression on the training set
  [w_xval(:, k), b_xval(k)] = LinearRegression(x_k, y_k, type);

  % Compute the estimate and the associated cross-validation NMSE
  nmse_xval(k) = RegressNMSE(w_xval(:, k), b_xval(k), x_xval, y_xval);
end

% Do the linear regression per se
[w, b, nmse, y_reg] = LinearRegression(x, y, type);

% Results on the cross-validations
fprintf(1, 'Average NMSE over 10-fold x-val: %g\n', mean(nmse_xval));
fprintf(1, 'Values of <w> found over the 10 x-val:\n');
disp(mean(w_xval, 2)');
fprintf(1, 'Values of <b> found over the 10 x-val:\n');
disp(mean(b_xval));

% Results on the full training data
fprintf(1, 'NMSE over trainning data: %g\n', nmse);
fprintf(1, 'Value of <w> on the training set:\n');
disp(w');
fprintf(1, 'Value of <b> on the training set:\n');
disp(b);


% -------------------------------------------------------------------------
function [w, b, nmse, y_reg] = LinearRegression(x, y, type)

% Do the regression
switch lower(type)
  case 'linear'
    % Mean square error linear regression
    x = [ones(1, length(y)); x];
    w = x' \ y';
    x = x(2:end, :);
  case 'lasso'
    [ind, w] = bolasso(x', y', 'statusBar', false, 'plotResults', false);
  otherwise
    error('Specify a type of regression: ''linear'', ''lasso'', etc...');
end

% Separate bias from weights
b = w(1);
w = w(2:end);

% Compute the estimate of y and the associated NMSE
[nmse, y_reg] = RegressNMSE(w, b, x, y);


% -------------------------------------------------------------------------
function [nmse, y_reg] = RegressNMSE(w, b, x, y)

y_reg = w' * x + b;
nmse = mean((y - y_reg).^2) / mean(y.^2);
