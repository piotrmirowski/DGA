% DGOA_Match  Match two cell arrays
%
% Syntax:
%   [ind_match0, ind_match1] = ...
%    DGOA_Match(raw_text0, raw_text1, col0, col1, raw0, raw1, date0, date1)

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

function [ind_match0, ind_match1] = ...
  DGOA_Match(raw_text0, raw_text1, col0, col1, raw0, raw1, date0, date1)

fprintf(1, 'Matching %s (negatives) with %s (positives)...\n', ...
  raw_text0{1, col0}, raw_text1{1, col1});
fprintf(1, 'Dates are in column %s (negatives) and %s (positives)...\n', ...
  raw_text0{1, date0}, raw_text1{1, date1});

n0 = size(raw_text0, 1) - 1;
n1 = size(raw_text1, 1) - 1;
ind_match0 = false(n0, 1);
ind_match1 = false(n1, 1);
for i = 1:n0
  j = 1;
  found = 0;
  while (~found && (j <= n1))
    if (isequal(raw_text0{i + 1, col0}, raw_text1{j + 1, col1}) && ...
        isequal(raw_text0{i + 1, date0}, raw_text1{j + 1, date1}))
      found = 1;
      ind_match0(i) = 1;
      ind_match1(j) = 1;
      if isequal(raw0(i, :), raw1(j, :))
        fprintf(1, 'Match: %s @ %d, %s @ %d\n', ...
          raw_text0{i + 1, col0}, i, raw_text1{j + 1, col1}, j);
      else
        fprintf(1, 'Error: %s @ %d, %s @ %d\n', ...
          raw_text0{i + 1, col0}, i, raw_text1{j + 1, col1}, j);
        disp(raw0(i, :));
        disp(raw1(j, :));
      end
    else
      j = j + 1;
    end
  end
end
fprintf(1, '\n');
  