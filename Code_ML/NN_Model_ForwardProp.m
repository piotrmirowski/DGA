% NN_Model_ForwardProp  Forward propagation through a single hidden-layer NN
%
% Syntax:
%   [yBar, z] = NN_Model_ForwardProp(x, model)
% Inputs:
%   x:       matrix of size <dim_x> x <n> of inputs
%   model:   struct containing the model
% Outputs:
%   yBar:    matrix of size <dim_y> x <n> of predictions
%   z:       matrix of size <dim_z> x <n> of hidden activations

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

function [yBar, z] = NN_Model_ForwardProp(x, model)

% Number of predictions (samples) which were made
n = size(x, 2);

% Use a stacked history and single A matrix to produce hidden activations
z = model.A * x + repmat(model.Abias, 1, n);

% Nonlinearity
z = tanh(z);

% Produce the prediction
yBar = model.B * z + repmat(model.Bbias, 1, n);
