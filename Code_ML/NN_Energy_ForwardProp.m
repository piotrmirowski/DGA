% NN_Energy_ForwardProp  Regression/classification measure of energy/error
%
% Syntax:
%   [e, perf] = NN_Energy_ForwardProp(y, yBar, params)
% Inputs:
%   y:      matrix of size <dim_x> x <n> of target outputs
%   yBar:   matrix of size <dim_y> x <n> of predicted outputs
%   params: struct containing the parameters
% Outputs:
%   e:      energy: Gaussian (regression) or logistic (classication)
%   perf:   (optional) measure of performance:
%           linear (Gaussian) regression: R2 correlation
%           logistic regression (classification): probability

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
% Version 1.0, New York, 18 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [e, perf] = NN_Energy_ForwardProp(y, yBar, params)

switch (params.energy)
  case 'gaussian'
    % (Twice the) energy for each sample
    dY2 = sum((y - yBar).^2, 1);
    % Total energy
    e = 0.5 * sum(dY2);
    
    if (nargout > 1)
      % R2 correlation
      mu = mean(y, 2);
      dYmu = sum((y - repmat(mu, 1, size(y, 2))).^2);
      perf = 1 - sum(dY2) / sum(dYmu);
    end
  case 'logistic'
    % Make the target class memberships <y> binary (0 or 1)
    y = round(y);
    % Avoid overflow errors
    yBar(yBar == 0) = eps;
    yBar(yBar == 1) = 1 - eps;
    % Compute the logistic negative log-likelihood
    e = - sum(y .* log(yBar) + (1 - y) .* log(1 - yBar), 1);
    % Total energy (negative log-likelihood)
    e = sum(e);

    if (nargout > 1)
      % Accuracy (in percentage)
      perf = 100 * mean(sum(round(yBar) == y, 2) / size(y, 2));
    end
end
