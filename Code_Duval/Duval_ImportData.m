% Self-explanatory function to import Duval data from a text file.

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
% Version 1.0, New York, 12 March 2010
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [mat, text] = Duval_ImportData(filename, n_cols, n_cols_text)

fid = fopen(filename, 'r');
i = 1;
j = 1;
while 1
  try
    line = fgetl(fid);
    if ~ischar(line), break; end
    if (j < 3)
      text{i, j} = line;
    else
      if isequal(line, '-')
        line = '0';
      end
      mat(i, j-2) = str2num(line);
    end
    j = j + 1;
    if (j > (n_cols + n_cols_text))
      j = 1;
      i = i + 1;
    end
  catch
    disp('error: debug...');
  end
end
fclose(fid);
