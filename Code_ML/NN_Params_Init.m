% NN_Params_Init  Initialize the default parameters
%
% Syntax:
%   params = NN_Params_Init()
% Outputs:
%   params:  struct containing the default parameters

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

function params = NN_Params_Init()

params = struct('dim_z', 10, ...
  'output', 'logistic', ...      % Classifier (logistic regression trained)
  'energy', 'logistic', ...      % Idem
  'len_batch', 1, ...            % Pure stochastic gradient
  'perc_xval', 0.1, ...          % Data are precious...
  'n_epochs', 1000, ...          % Max bound (as there is early stopping)
  'eta', 0.1, ...                % eta=1 jumps too much, 0.1 might be good
  'lambda', 1e-4, ...            % 1e-4 is a good default value
  'momentum', 0.01, ...          % Small momentum
  'eta_anneal', 0.99, ...        % Decrease by 2/3 after 100 epochs
  'verbose', 1);                 % Level of verbosity
