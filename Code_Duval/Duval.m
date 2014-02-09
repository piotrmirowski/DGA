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

% Duval limits
lim_duval = ...
  [log10(100); ...  % H2
   log10(1e9); ...  % CH4
   log10(300); ...  % C2H2
   log10(100); ...  % C2H4
   log10(20); ...   % C2H6
   log10(1e9); ...  % CO
   log10(1e9)];     % CO2

% How many times shall we run the experiments
if ~exist('indJ', 'var')
  indJ = 1;
end

% Ratio of negative vs. positive samples
frac_neg = [0.3 0.4 0.5 0.6 0.7];

% Percentage of data being cross-validated
thres_xval = 0.8;

% Main loop over the experiments and ratios of samples
for j = indJ
  for i = length(frac_neg):(-1):1
    f = frac_neg(i);

    % Prepare the data
    [filename_raw, filename_proc, dgoa, labels, gasNames] = ...
      Duval_PrepareData(path_data, (i - 1) * 10 + j, f, 0);

    % Run the classification using Duval thresholds
    labelsDuval = zeros(size(labels));
    for k = 1:7
      labelsDuval = labelsDuval + (dgoa(k, :) > lim_duval(k));
    end
    accuracy_Duval(i, j) = ...
      sum((labelsDuval>0) == labels) / length(labels) * 100;

    % Run the classification using several ML techniques
    res = Duval_EvaluateML(dgoa, labels, thres_xval);

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
    save(['Duval_Statistics_' num2str(j) '_' num2str(f) '.mat']);
  end
end


