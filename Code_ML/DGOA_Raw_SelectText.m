% DGOA_Raw_SelectText  Select certain rows in a text cell array
%
% Syntax:
%   raw_text_filt = DGOA_Raw_SelectText(raw_text, ind)
% Inputs:
%   raw_text:      cell array of size <N+1> x <K> with the content
%                  of the full speadsheet
%   ind:           boolean vector of length <N>
% Outputs:
%   raw_text:      cell array of size <N'+1> x <K> with the content
%                  of the full speadsheet, where <N'> <= <N>

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
% Version 1.0, New York, 27 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function raw_text_filt = DGOA_Raw_SelectText(raw_text, ind)

n_rows = size(raw_text, 1) - 1;
n_cols = size(raw_text, 2);
raw_text_filt = cell(n_rows - sum(ind), n_cols);
m = 0;
for k = 1:n_rows
  if ~ind(k)
    m = m + 1;
    for i = 1:n_cols
      raw_text_filt{m+1, i} = raw_text{k+1, i};
    end
  end
  if (mod(k, 1000) == 0), fprintf(1, '.'); end
end
fprintf(1, '\n');
