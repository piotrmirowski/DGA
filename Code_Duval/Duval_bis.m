% Main script to analyze Duval data using a battery of ML algorithms.

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

path_this = pwd;
path_data = [path_this '/../Data_Duval/'];
addpath('../Data_Duval/');
addpath('../Code_ML/');
addpath('../../libs/netlab/');

for j = indJ
  for i = length(frac_neg):(-1):1
    f = frac_neg(i);

    % Save the data and results
    load(['Duval_Statistics_' num2str(j) '_' num2str(f) '.mat']);

    % Run the classification using several additional ML techniques
    res = Duval_EvaluateML_bis(dgoa, labels, thres_xval, res);

    % Extract statistics (mean and std)
    statNames = fieldnames(res);
    for k = 1:length(statNames)
      try
        nam = statNames{k};
        cmd = sprintf('%s_mean(%d, %d) = mean(res.%s);', nam, i, j, nam);
        eval(cmd);
        cmd = sprintf('%s_std(%d, %d) = std(res.%s);', nam, i, j, nam);
        eval(cmd);
      end
    end
    
    % Save the results
    save(['Duval_Statistics_bis_' num2str(j) '_' num2str(f) '.mat']);
  end
end


