% Self-explanatory function to save Duval results to a text file.

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

function res = Duval_SaveTxt(filename)

load([filename '.mat']);

f = fopen([filename '.txt'], 'w');

statNames = fieldnames(res);
for k = 1:length(statNames)
  try
    nam = statNames{k};
    cmd = sprintf('val_mean = mean(res.%s);', nam);
    eval(cmd);
    cmd = sprintf('val_std = std(res.%s);', nam);
    eval(cmd);
    fprintf(f, '%s,mean,%g\n', nam, val_mean);
    fprintf(f, '%s,std,%g\n', nam, val_std);
  end
end

fclose(f);
