% Duval_EvaluateML_bis  Evaluate many ML algorithms on data
%
% Syntax:
%   res = Duval_EvaluateML_bis(dgoa, labels, thres_xval, res)
% Inputs:
%   dgoa:          matrix of size 7 x <N> of processed DGA data
%   labels:        vector of length <N> of labels (0=faulty, 1=normal)
%   thres_xval:    cross-validation threshold (typically 0.8)
%   res:           struct containing the results
% Outputs:
%   res:           struct containing the results

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

function res = Duval_EvaluateML_bis(dgoa, labels, thres_xval, res)


dgoaNorm = DGOA_Inputs_Standardize(dgoa, []);
n_val = 10;
n_samples = size(dgoaNorm, 2);
if (nargin < 4), res = []; end


% Try classifying using low density separation
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_LDS(k), yTestProb, res.C_LDS(k), res.rho_LDS(k)] = ...
    LDS_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind));

  % Evaluate the ROC and AUC
  [res.tpr_LDS{k}, res.fpr_LDS{k}, res.auc_LDS(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'LDS: accuracy=%.4f (+/-%.4f)\n', ...
  mean(res.accuracy_LDS), std(res.accuracy_LDS));


% Try classifying using local linear semi-supervised regression
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.r2_LLSSR(k), yTestProb, res.bandwidth_LLSSR(k)] = ...
    LLSSR_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 1, 1);

  % Evaluate the ROC and AUC
  [res.tpr_LLSSR{k}, res.fpr_LLSSR{k}, res.auc_LLSSR(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'LLSSR: R2=%.4f (+/-%.4f)\n', ...
  mean(res.r2_LLSSR), std(res.r2_LLSSR));


% -------------------------------------------------------------------------
function [tpr, fpr, auc] = EvaluateROC(yTestProbMat, labels)

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

