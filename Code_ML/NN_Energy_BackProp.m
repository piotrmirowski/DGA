% NN_Learn  (Stochastically) learn the model parameters on a mini-batch
%
% Syntax:
%   dL_dy = NN_Energy_BackProp(y, yBar, params)
% Inputs:
%   y:      matrix of size <dim_x> x <n> of target outputs
%   yBar:   matrix of size <dim_y> x <n> of predicted outputs
%   params: struct containing the parameters
% Outputs:
%   e:      energy
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

function dL_dy = NN_Energy_BackProp(y, yBar, params)

switch (params.energy)
  case 'gaussian'
    % Derivative of the Gaussian energy w.r.t. yBar
    dL_dy = yBar - y;
  case 'logistic'
    % % 1) Logistic regression
    % %    Derivative of the classification energy w.r.t. classifier output
    % dL_dyBar = -y ./ yBar + (1 - y) ./ (1 - yBar);
    % % 2) Jacobian of the logistic transfer on the code w.r.t. input sum
    % dLogistic_dsum = yBar .* (1 - yBar);
    % % 3) Derivative of the classification energy w.r.t. the linear sum
    % dL_dy = dL_dyBar * dLogistic_dsum;
    % But the above could be written:
    % dL_dy = ...
    %   (yBar - y) ./ (yBar .* (1 - yBar)) .* yBar .* (1 - yBar)
    dL_dy = yBar - y;
end
