% NN_Learn  (Stochastically) learn the model parameters on a mini-batch
%
% Syntax:
%   [e, model] = NN_Learn(x, y, model, params)
% Inputs:
%   x:      matrix of size <dim_x> x <n> of inputs
%   y:      matrix of size <dim_y> x <n> of target outputs
%   model:  struct containing the model after 1 learning step
%   params: struct containing the parameters
% Outputs:
%   e:      vector of length <n> of energies
%   model:  struct containing the model after 1 learning step

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

function [e, model] = NN_Learn(x, y, model, params)

% Forward propagation to get the prediction (and hidden activations)
[yBar, z] = NN_Model_ForwardProp(x, model);
if isequal(params.output, 'logistic')
  yBar = NN_Logistic(yBar);
end

% Evaluate the energy (regression or classification)
e = NN_Energy_ForwardProp(y, yBar, params);

% Evaluate the gradient of the regression/classification energy (loss)
dL_dy = NN_Energy_BackProp(y, yBar, params);

% Compute the gradients to the model
model = NN_Model_BackProp(x, z, dL_dy, model);

% Gradient-based learning with momentum and regularization
model = ApplyGradients(model, params);


% -------------------------------------------------------------------------
function model = ApplyGradients(model, params)

% Momentum term on the dynamical parameters
momentum = params.momentum;
if (momentum > 0)

  % 1) Add momentum term
  if isfield(model, 'dL_dbA_prev')
    model.dL_dA = ...
      model.dL_dA * (1 - momentum) + model.dL_dA_prev * momentum;
    model.dL_dbA = ...
      model.dL_dbA * (1 - momentum) + model.dL_dbA_prev * momentum;
    model.dL_dB = ...
      model.dL_dB * (1 - momentum) + model.dL_dB_prev * momentum;
    model.dL_dbB = ...
      model.dL_dbB * (1 - momentum) + model.dL_dbB_prev * momentum;
  end

  % 2) Store the gradients for next iteration
  model.dL_dA_prev = model.dL_dA;
  model.dL_dbA_prev = model.dL_dbA;
  model.dL_dB_prev = model.dL_dB;
  model.dL_dbB_prev = model.dL_dbB;
end
  
% L1-norm regularization on the matrix weights
lambda = params.lambda;
model.dL_dA = model.dL_dA + lambda * sign(model.dL_dA);
model.dL_dB = model.dL_dB + lambda * sign(model.dL_dB);

% Gradient descent
model.A = model.A - params.eta * model.dL_dA;
model.Abias = model.Abias - params.eta * model.dL_dbA;
model.B = model.B - params.eta * model.dL_dB;
model.Bbias = model.Bbias - params.eta * model.dL_dbB;
