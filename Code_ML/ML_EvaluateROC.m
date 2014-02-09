% ML_EvaluateROC  Compute the ROC curve and the AUC
%
% Syntax:
%   [tpr, fpr, auc] = ML_EvaluateROC(yTestProbMat, labels)
% Inputs:
%   yTestProbMat: vector of length <n> of predicted label probabilities
%   labels:       vector of length <n> of true labels (binary)
% Outputs:
%   tpr:  vector of true positive rates going from 0 to 1
%   fpr:  vector of false positive rates corresponding to tpr values
%   auc:  area under the precision-recall curve

% Copyright (C) 2010 Piotr Mirowski
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
% Version 1.0, New York, 24 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function [tpr, fpr, auc] = ML_EvaluateROC(yTestProbMat, labels)

% Retrieve bad transformers in the order
if (numel(yTestProbMat) == length(yTestProbMat))
  yTestProb = yTestProbMat;
else
  yTestProb = yTestProbMat(:, 1);
  accuracy2 = sum((yTestProb >= 0.5)' == labels(:)') / length(labels) * 100;
  if (accuracy2 < 50)
    yTestProb = yTestProbMat(:, 2);
  end
end
[dummy, order] = sort(yTestProb, 'descend');
predictions = labels(order);

% Compute ROC curve
n_pos = sum(labels);
[tpr, fpr] = IE_ROC(predictions, 1, n_pos, 0);

% Compute AUC
auc = sum(diff(fpr) .* (tpr(1:(end-1)) + tpr(2:end)) / 2);

% Fix little bug with ROC curve when all predictions are positive
if (length(unique(yTestProb)) == 1)
  auc = 0.5;
end
