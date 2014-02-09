% DGOA_ExtractCommonVariables  Extract common variables from 2 sets of data
%
% Syntax:
%   [xPos, yNeg, labels, labelsPos, labelsNeg] = ...
%     DGOA_ExtractCommonVariables(raw_positives, raw_positives_text, ...
%     raw_negatives, raw_negatives_text)
% Inputs (variables that need to be loaded in memory):
%   raw_positives:      Matrix of size <nPos> x <nVarPos>
%   raw_positives_text: Cell array of size <nPos+1> x <n>
%   raw_negatives:      Matrix of size <nNeg> x <nVarNeg>
%   raw_negatives_text: Cell array of size <nNeg+1> x <n>
% Ouputs:
%   x_pos:              Matrix of size <nPos> x <nVar>
%   x_neg:              Matrix of size <nNeg> x <nVar>
%   labels:             Cell array of length <nVar>
%   labelsPos:          Cell array of length <nVarPos>
%   labelsNeg:          Cell array of length <nVarNeg>

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

function [x_pos, x_neg, labels, labelsPos, labelsNeg] = ...
  DGOA_ExtractCommonVariables(raw_positives, raw_positives_text, ...
  raw_negatives, raw_negatives_text)

% Prepare the positive and negative DG data
nVarPos = size(raw_positives, 2);
nPos = size(raw_positives, 1);
if (size(raw_positives_text, 1) ~= (nPos + 1))
  error('Cell array <raw_positives_text> not of size <nPos+1> x <n>');
end
nVarNeg = size(raw_negatives, 2);
nNeg = size(raw_negatives, 1);
if (size(raw_negatives_text, 1) ~= (nNeg + 1))
  error('Cell array <raw_negatives_text> not of size <nNeg+1> x <n>');
end

% Create the labels for positive data from the last column names in 
% <raw_positives_text>
labelsPos = cell(1, nVarPos);
n = size(raw_positives_text, 2);
for k = 1:nVarPos
  j = k + n - nVarPos;
  labelsPos{k} = raw_positives_text{1, j};
end
fprintf(1, 'Labels of the positive data:\n');
for k = 1:nVarPos
  fprintf(1, '%d: %s\n', k, labelsPos{k});
end
fprintf(1, '\n');

% Create the labels for negative data from the last column names in
% <raw_negatives_text>
labelsNeg = cell(1, nVarNeg);
n = size(raw_negatives_text, 2);
for k = 1:nVarNeg
  j = k + n - nVarNeg;
  labelsNeg{k} = raw_negatives_text{1, j};
end
fprintf(1, 'Labels of the negative data:\n');
for k = 1:nVarNeg
  fprintf(1, '%d: %s\n', k, labelsNeg{k});
end
fprintf(1, '\n');

% Match the two labels and select only common channels
k = 0;
indPos = [];
indNeg = [];
labels = {};
for i = 1:nVarPos
  j = 1;
  while (j <= nVarNeg)
    if isequal(labelsPos{i}, labelsNeg{j})
      indPos = [indPos i];
      indNeg = [indNeg j];
      k = k + 1;
      labels{k} = labelsPos{i};
      labels{k} = strrep(labels{k}, '"', '');
      fprintf(1, 'Matched channel %d: %s in positives and negatives\n', ...
        k, labels{k});
      break;
    else
      j = j + 1;
    end
  end
end
fprintf(1, '\n');

% Select the positive and negative data and define <x_pos> and <x_neg>
x_pos = raw_positives(:, indPos)';
x_neg = raw_negatives(:, indNeg)';

