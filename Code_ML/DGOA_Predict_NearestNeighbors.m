% DGOA_Predict_NearestNeighbors  Perform nearest-neighbors prediction
%
% Syntax:
%   y_neg_nn = ...
%     DGOA_Predict_NearestNeighbors(x_pos, y_pos, x_neg, labels, gamma)
% Inputs:
%   x_pos: matrix of positive inputs of size <D> x <N>
%   y_pos: vector of target outputs of size 1 x <N>
%   x_neg: matrix of positive inputs of size <D> x <M>
%   gamma: spread of the Gaussian kernel for the NN weights
% Outputs:
%   y_neg: vector of interpolated outputs of size 1 x <N>,
%          using the nearest neighbors algorithm

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
% Version 1.0, New York, 27 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function y_neg_nn = ...
  DGOA_Predict_NearestNeighbors(x_pos, y_pos, x_neg, labels, gamma)

% Standardize the input data to unit variance and zero mean
[x_pos_norm, x_neg_norm] = DGOA_Inputs_Standardize(x_pos, x_neg);

% Scale the target output data to [0, 1]
y_min = min(y_pos);
y_max = max(y_pos);
y_pos_scale = (y_pos - y_min) / (y_max - y_min);

% Plot the positive dataset
[x_pos_sel, labels_sel] = DGOA_SelectTriple(x_pos, [2 6 7], labels);
DGOA_PlotDGOA3D(x_pos_sel, y_pos_scale, labels_sel);
title('Time to problem (positives): now (red) to later (green)');
plot3(x_neg(2, :), x_neg(6, :), x_neg(7, :), 'b.', 'MarkerSize', 1);

% Cross-validation on the training dataset to find the hyperparameter gamma
if (nargin < 5)
  gamma = FindGamma(x_pos_norm, y_pos_scale);
end

% Compute the distance matrix between positive and negative data
fprintf(1, 'Computing Gaussian nearest-neighbors with gamma=%g...\n', ...
  gamma);
y_neg_nn = NearestNeighbors(x_pos_norm, y_pos_scale, x_neg_norm, gamma);

% Plot the nearest-neighbors interpolation
fprintf(1, 'Plotting the nearest-neighbors interpolation...\n');
x_neg_sel = DGOA_SelectTriple(x_neg, [2 6 7], labels);
DGOA_PlotDGOA3D(x_neg_sel, y_neg_nn, labels_sel);
title('Time to problem (NN interpolation): now (red) to later (green)');

% Rescale the interpolated output
y_neg_nn = y_neg_nn * (y_max - y_min) + y_min;


% -------------------------------------------------------------------------
function [gamma_best, nmse_best] = FindGamma(x, y)

% Create random selection of samples
n_samples = length(y);
n_samples_test = ceil(n_samples / 10);
[dummy, ind_random] = sort(rand(1, n_samples));
x = x(:, ind_random);
y = y(ind_random);

% Perform a grid search
nmse_best = inf;
for gamma =  2.^[-15:0.5:5]

  % 10-fold cross-validation
  nmse_vec = zeros(1, 10);
  k = 0;
  for i = 1:n_samples_test:n_samples
    k = k + 1;
    % Select cross-validation samples
    indKxVal = i:(i+n_samples_test-1);
    indKxVal = unique(min(indKxVal, n_samples));
    % Select training samples
    indKtrain = setdiff(1:n_samples, indKxVal);
    % Select training input data
    x_ak = x(:, indKtrain);
    % Select cross-validation input data
    x_bk = x(:, indKxVal);
    % Select output training data
    y_ak = y(indKtrain);
    % Select output cross-validation data
    y_bk = y(indKxVal);
    
    % Predict the cross-validation data using NN
    y_bk_nn = NearestNeighbors(x_ak, y_ak, x_bk, gamma);
    
    % Estimate the error
    nmse_vec(k) = mean((y_bk - y_bk_nn).^2) / mean(y_bk.^2);
    fprintf(1, '.');
  end
  
  % Average cross-validation error
  nmse = mean(nmse_vec);
  fprintf(1, '\ngamma=%g, x-val NMSE=%.4f\n', gamma, nmse);
  if (nmse < nmse_best)
    nmse_best = nmse;
    gamma_best = gamma;
  end
end
  
  
% -------------------------------------------------------------------------
function y_b_nn = NearestNeighbors(x_a, y_a, x_b, gamma)

len_a = size(x_a, 2);
len_b = size(x_b, 2);

% Compute the distance matrix between positive and negative data
dist_ab = repmat(sum(x_b.^2)', 1, len_a) + ...
  repmat(sum(x_a.^2), len_b, 1) - 2 * x_b' * x_a;
dist_ab = max(dist_ab, 0);

% Convert the distance matrix to a weight matrix using a Gaussian kernel
w_ab = exp(-gamma * dist_ab);
y_b_nn = (y_a * w_ab') ./ sum(w_ab, 2)';

