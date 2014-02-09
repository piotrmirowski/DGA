% Duval_PrepareData  Prepare a Duval dataset with additional negative data
%
% Syntax:
%   [filename_raw, filename_proc, dgoa, labels, gasNames] = ...
%     Duval_PrepareData(path_data, file_index, frac_neg, do_display)
% Inputs:
%   path_data:  path to the text file containing Duval data
%   file_index: number to be appended to the new file name
%   frac_neg:   fraction of "negatives"
%   do_display: 1 is we want to plot
% Outputs:
%   filename_raw:  filename for the raw data
%   filename_proc: filename for the log-normalized processed data
%   dgoa:          matrix of size 7 x <N> of processed DGA data
%   labels:        vector of length <N> of labels (0=faulty, 1=normal)
%   gasNames:      array of cells containing the names of each gas

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

function [filename_raw, filename_proc, dgoa, labels, gasNames] = ...
  Duval_PrepareData(path_data, file_index, frac_neg, do_display)


% Faulty transfomers
[duvalPD, duvalPD_text] = ...
  Duval_ImportData([path_data 'DuvalTable2.txt'], 8, 2);
[duvalD1, duvalD1_text] = ...
  Duval_ImportData([path_data 'DuvalTable3.txt'], 8, 2);
[duvalD2, duvalD2_text] = ...
  Duval_ImportData([path_data 'DuvalTable4.txt'], 8, 2);
[duvalT1T2, duvalT1T2_text] = ...
  Duval_ImportData([path_data 'DuvalTable5.txt'], 8, 2);
[duvalT3, duvalT3_text] = ...
  Duval_ImportData([path_data 'DuvalTable6.txt'], 8, 2);

% Normal transformers
[duvalNormalNoOLTC, duvalNormalNoOLTC_text] = ...
  Duval_ImportData([path_data 'DuvalAnnex2Table1.txt'], 8, 2);
[duvalNormalOLTC, duvalNormalOLTC_text] = ...
  Duval_ImportData([path_data 'DuvalAnnex2Table2.txt'], 8, 2);
[duvalNormalInst, duvalNormalInst_text] = ...
  Duval_ImportData([path_data 'DuvalAnnex2Table4.txt'], 8, 2);

% Normal transformers
duvalNormalNoOLTCrange = ...
  [60, 150; 40, 110; 3, 50; 60, 280; 50, 90; 540, 900; 5100, 13000; 10, 10]';
duvalNormalOLTCrange = ...
  [75, 150; 35, 130; 80, 270; 110, 250; 50, 70; 400, 850; 5300, 12000; 10, 10]';
duvalNormalInstRange = ...
  [6, 300; 11, 120; 1, 5; 3, 40; 7, 130; 250, 1100; 800, 4000; 10, 10]';

% Gather positive and negative data so far
duvalPos = [duvalPD; duvalD1; duvalD2; duvalT1T2; duvalT3]';
duvalNeg = [duvalNormalNoOLTC; duvalNormalOLTC; duvalNormalInst]';
n_pos = size(duvalPos, 2);
n_neg = size(duvalNeg, 2);

% Normal transformers (uniform sampling from range values)
% Since there are 117 faulty samples and 50 normal ones, we generate
% 66 additional normal samples in order to balance the datasets
n_neg_rand = (n_pos * frac_neg + n_neg * (frac_neg - 1)) / (1 - frac_neg);
n = round(n_neg_rand / 3);
duvalNormalNoOLTCrand = Duval_GenerateData(duvalNormalNoOLTCrange, n);
duvalNormalOLTCrand = Duval_GenerateData(duvalNormalOLTCrange, n);
duvalNormalInstRand = Duval_GenerateData(duvalNormalInstRange, n);

% Add the generated negative data
duvalNeg = [duvalNeg, ...
  duvalNormalNoOLTCrand', duvalNormalOLTCrand', duvalNormalInstRand'];

% Remove last row and set column names
duvalPos = duvalPos(1:7, :);
duvalNeg = duvalNeg(1:7, :);
gasNames = {'H_2', 'CH_4', 'C_2H_2', 'C_2H_4', 'C_2H_6', 'CO', 'CO_2'};

% Save the dataset
filename_raw = sprintf('%sDuval_Raw_%d.mat', path_data, file_index);
save(filename_raw);


% Prepare data for the classifier
dgoa = [duvalPos duvalNeg];
n_pos = size(duvalPos, 2);
n_neg = size(duvalNeg, 2);
labels = [ones(1, n_pos) zeros(1, n_neg)];
[dgoa, gasNames] = DGOA_Inputs_LogNormalize(dgoa, 'log10', gasNames);

filename_proc = ...
  sprintf('%sDuval_Classification_%d.mat', path_data, file_index);
save(filename_proc, 'dgoa', 'labels', 'gasNames');


if (do_display)
  % Visualize the data in 3D
  classNames = {'Normal', 'Fault'};
  DGOA_PlotDGOA3D(dgoa([1 6 7], :), labels+1, gasNames([1 6 7]), ...
    classNames, 'Duval');
  DGOA_PlotDGOA3D(dgoa([3:5], :), labels+1, gasNames([3:5]), ...
    classNames, 'Duval');
  DGOA_PlotDGOA3D(dgoa([1 2 4], :), labels+1, gasNames([1 2 4]), ...
    classNames, 'Duval');

  % indNonRand = [1:(117+50)];
  % DGOA_PlotDGOA3D(dgoa([1 6 7], indNonRand), labels(indNonRand)+1, ...
  %   gasNames([1 6 7]), classNames, 'Duval');
  % DGOA_PlotDGOA3D(dgoa([3:5], indNonRand), labels(indNonRand)+1, ...
  %   gasNames([3:5]), classNames, 'Duval');
  % DGOA_PlotDGOA3D(dgoa([1 2 4], indNonRand), labels(indNonRand)+1, ...
  %   gasNames([1 2 4]), classNames, 'Duval');
end
