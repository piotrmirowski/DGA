% LLSSR_Predict  Non-parametric local linear semi-supervised regression
%
% Syntax:
%   yB = LLR_Predict(xLabeled, yLabeled, xUnlabeled, h, gamma)
% Inputs:
%   xLabeled:   matrix of size <dim_x> x <n_a> of labeled inputs
%   yLabeled:   matrix of size <dim_y> x <n_a> of labeled outputs
%   xUnlabeled: matrix of size <dim_x> x <n_b> of unlabeled inputs
%   h:          bandwidth parameter for the Gaussian kernel (default: 1)
%   gamma:      regularization parameter (default: 1)
% Outputs:
%   yLabeled: matrix of size <dim_y> x <n_b> of predicted (unlabeled) outputs
%
% Reference:
%   M. R. Rwebangira and J. Lafferty,
%   "Local linear semi-supervised regression",
%   School of Computer Science Carnegie Mellon University, 
%   Pittsburgh, PA 15213, Tech. Rep. CMU-CS-09-106, Feb. 2009

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

function yUnlabeledBar = ...
  LLSSR_Predict(xLabeled, yLabeled, xUnlabeled, h, gamma)

warning off;

n_a = size(xLabeled, 2);
n_b = size(xUnlabeled, 2);
n = n_a + n_b;
d = size(xLabeled, 1);

% Kernel matrix on the labeled and unlabeled data
x = [xLabeled xUnlabeled];
dis = repmat(sum(x.^2, 1)', 1, n) + repmat(sum(x.^2, 1), n, 1) - 2 * (x') * x;
w = 1 / sqrt(2 * pi) * exp(-0.5 / h^2 * dis);

% Randomly initialize the local linear regression weights
B = randn(n, d+1) * d^(-1/2);
b0_prev = zeros(n, 1);
zeroMat = zeros(d+1);
zeroVec = zeros(d+1, 1);

converged = 0;
epoch = 1;
max_epochs = 100;
e_conv = 1e-3;
while ~converged

  % Recompute coefficients for all samples in the labeled and unlabeled set
  order = randperm(n);
  for i = order
    Xii = [1; x(:, i) - x(:, i)];
    XjiMat = [ones(1, n); zeros(d, n)];
    XijMat = [ones(1, n); zeros(d, n)];
    for k = 1:d
      XjiMat(k+1, :) = x(k, :) - x(k, i);
      XijMat(k+1, :) = x(k, i) - x(k, :);
    end
    wi = w(:, i);
    wi_lab = w(1:n_a, i);
    Xji_lab = XjiMat(:, 1:n_a);
    wi_unlab = w((n_a+1):end, i);
    Xji_unlab = XjiMat(:, (n_a+1):end);
    
    term1 = (Xji_lab .* repmat(wi_lab', d+1, 1)) * Xji_lab';
    term2 = zeros(d+1);
    term2(1) = sum(w(:, i));
    term5 = (Xji_unlab .* repmat(wi_unlab', d+1, 1)) * Xji_unlab';
    term3 = Xji_lab * (wi_lab .* yLabeled');
    term4 = Xii * sum(XijMat .* B') * wi;
    term6 = XjiMat * (wi .* B(:,1));

    B(i, :) = (term1 + gamma * term2 + gamma * (term1 + term5)) \ ...
      (term3 + gamma * (term4 + term6));
  end

  % Evaluate the convergence
  b0 = B(:, 1);
  e = norm(b0 - b0_prev) / norm(b0);
  b0_prev = b0;
  if ((e < e_conv) || isnan(e) || (epoch >= max_epochs))
    converged = 1;
  end
  fprintf(1, '.');
 
%   figure(1);
%   clf;
%   hold on;
%   plot(xLabeled, yLabeled, 'r.');
%   plot([xLabeled xUnlabeled], b0, 'b.');
  
%   figure(1);
%   clf;
%   plot(yLabeled, 'b.');
%   hold on;
%   plot(b0, 'r.');
  
  epoch = epoch + 1;
end
fprintf(1, '\nConverged in %d epochs with delta_e=%g\n', epoch, e);


% Make the predictions
yUnlabeledBar = b0((n_a+1):end)';
