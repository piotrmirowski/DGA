% NN_Model_BackProp  Back-propagation: compute the gradients to the model
%
% Syntax:
%   model = NN_Model_BackProp(x, z, dL_dy, model)
% Inputs:
%   x:     matrix of size <dim_z> x <n> of inputs
%   z:     matrix of size <dim_h> x <n> of hidden activations
%   dL_dy: matrix of size <dim_y> x <n> of gradient on the predictions
%   model: struct containing the model
% Outputs:
%   model: struct containing the model with updated gradients

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

function model = NN_Model_BackProp(x, z, dL_dy, model)

% Derivatives w.r.t. hidden activations
dL_dz = model.B' * dL_dy;

% Derivative w.r.t. linear transformation matric B (second layer)
model.dL_dB = dL_dy * z';
  
% Derivatives w.r.t. bias of the second layer
model.dL_dbB = sum(dL_dy, 2);

% Derivatives w.r.t. tanh nonlinearity
dL_dsum = dL_dz .* (1 - z.^2);

% Derivative w.r.t. linear transformation matric A
model.dL_dA = dL_dsum * x';

% Derivatives w.r.t. bias
model.dL_dbA = sum(dL_dsum, 2);
