% IE_CompareClasses  Compare discrete labels or probability distributions
%
% Syntax:
%   res = IE_CompareClasses(classes1, class2)
% Inputs:
%   classes: matrix of classes of size <K> x <T> (<K> can be 1)
%   class2:  scalar or vector of classes of size <K> x 1
% Outputs:
%   res:     boolean vector of class similarities/overlap

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
% Version 1.0, New York, 18 July 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function res = IE_CompareClasses(classes1, class2)

if (size(class2, 2) > 1)
  error('<class2> has more than 1 column');
end

if ((size(classes1, 1) == 1) && (numel(class2) == 1))
  % Comparing discrete labels
  res = (classes1 == class2);
elseif ((size(classes1, 1) == size(class2, 1)))
  n_samples_1 = size(classes1, 2);
  res = classes1 & repmat(class2, 1, n_samples_1);
  res = min(sum(res), 1);
else
  error('Inconsistent dimensions in <classes1> and <class2>');
end
  