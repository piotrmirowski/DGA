% NN_Model_Init  Create a neural network (single hidden-layer) model
%
% Syntax:
%   model = NN_Model_Init(params)
% Inputs:
%   params:  struct containing the parameters
% Outputs:
%   model:   struct containing the model

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

function model = NN_Model_Init(params)

if (params.verbose > 1)
  fprintf(1, 'Initializing the NN model...\n');
end

% Dimensions of the modules
dim_x = params.dim_x;
dim_z = params.dim_z;
dim_y = params.dim_y;

% Basic components of the model
model = struct('dim_x', dim_x, 'dim_z', dim_z, 'dim_y', dim_y, ...
  'output', params.output, ...                 % 'linear' or 'logistic'
  'measure', inf, ...                          % For tracking best perf
  'A', [], 'Abias', [], 'B', [], 'Bbias', []); % NN weights

% Single dynamics
model.A = randn(dim_z, dim_x) * dim_x^(-1/2);
model.Abias = zeros(dim_z, 1);
model.B = randn(dim_y, dim_z) * dim_z^(-1/2);
model.Bbias = zeros(dim_y, 1);
