% DGOA_SelectTriple  Select a triple of DGOA variables
%
% Syntax:
%   [x_sel, labels_sel] = DGOA_SelectTriple(x, ind_dim, labels)
% Inputs:
%   x:       matrix of size <D> x <N> of <D> DG variables and <N> samples
%   ind_dim: vector of dimension indexes in <x> to be plotted
%   labels:  cell array of char string with names of the variables

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
% Version 1.0, New York, 23 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [x_sel, labels_sel] = DGOA_SelectTriple(x, ind_dim, labels)

if (length(ind_dim) ~= 3)
  error('You need to select 3 dimensions in <ind_dim>');
end
x_sel = x(ind_dim, :);
labels_sel = cell(1, 3);
for k = 1:3
  labels_sel{k} = labels{ind_dim(k)};
end
