% DGOA_ExtractClass
%
% Script that prepares the dataset:
% * transforms raw DGOA data into inputs features
% * prepares the target values on the "positive" dataset
% * does a few simple statistics
%
% Syntax:
% function [class_idx, class_names, class_type] = ...
%   DGOA_ExtractClass(raw_text, col)
% Inputs:
%   raw_text:    Cell array of size <n> x <nVar> of raw text data
%   col:         Column index
% Ouputs:
%   class_idx:   Vector of class indexes of length <n>, values from 1 to <K>
%   class_names: Cell array of length <K> with class names
%   class_type:  Label of the classification

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

function [class_idx, class_names, class_type] = ...
  DGOA_ExtractClass(raw_text, col)

class_type = raw_text{1, col};
n = size(raw_text, 1) - 1;
class_idx = zeros(1, n);
n_labels = 0;
for k = 1:n
  lab_k = raw_text{k+1, col};
  i = 0;
  for j = 1:n_labels
    if isequal(lab_k, class_names{j})
      i = j;
    end
  end
  if (i == 0)
    n_labels = n_labels + 1;
    class_names{n_labels} = lab_k;
    class_idx(k) = n_labels;
  else
    class_idx(k) = i;
  end
end
