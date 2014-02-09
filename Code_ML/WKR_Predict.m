% WKR_Predict  Non-parametric weighted kernel regression (Nadaraya-Watson)
%
% Syntax:
%   yB = WKR_Predict(xA, yA, xB, h)
% Inputs:
%   xA: matrix of size <dim_x> x <n_a> of labeled inputs
%   yA: matrix of size <dim_y> x <n_a> of labeled outputs
%   xB: matrix of size <dim_x> x <n_b> of unlabeled inputs
%   h:  bandwidth parameter for the Gaussian kernel
% Outputs:
%   yA: matrix of size <dim_y> x <n_b> of predicted (unlabeled) outputs
%
% Reference:
%   Nadaraya, E. A. (1964). "On Estimating Regression".
%   Theory of Probability and its Applications 9 (1): 141-142

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

function yB = WKR_Predict(xA, yA, xB, h)

% Compute the distance matrix between train and test data
n_a = size(xA, 2);
n_b = size(xB, 2);
distAB = repmat(sum(xB.^2)', 1, n_a) + ...
  repmat(sum(xA.^2), n_b, 1) - 2 * xB' * xA;
distAB = max(distAB, 0);

% Convert the distance matrix to a weight matrix using a Gaussian kernel
wAB = 1 / sqrt(2 * pi) * exp(-0.5 / h^2 * distAB);

% Predict using the labelled data and the Gaussian kernel
yB = (yA * wAB') ./ sum(wAB, 2)';
