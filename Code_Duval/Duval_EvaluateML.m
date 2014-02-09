% Duval_EvaluateML  Evaluate many ML algorithms on data
%
% Syntax:
%   res = Duval_EvaluateML(dgoa, labels, thres_xval, res)
% Inputs:
%   dgoa:          matrix of size 7 x <N> of processed DGA data
%   labels:        vector of length <N> of labels (0=faulty, 1=normal)
%   thres_xval:    cross-validation threshold (typically 0.8)
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

function res = Duval_EvaluateML(dgoa, labels, thres_xval)


dgoaNorm = DGOA_Inputs_Standardize(dgoa, []);
n_val = 10;
n_samples = size(dgoaNorm, 2);


% Try classifying using linear regression and the AUC
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [w, b] = ...
    DGOA_LinearRegression(dgoaNorm(:,~ind), labels(~ind), 'linear');
  yTestProb = w' * dgoaNorm(:,ind) + b;
  y = labels(ind);
  res.r2_LinReg(k) = 1 - sum((yTestProb - y).^2) / sum((y - mean(y)).^2);
  
  % Evaluate the ROC and AUC
  [res.tpr_LinReg{k}, res.fpr_LinReg{k}, res.auc_LinReg(k)] = ...
    EvaluateROC(yTestProb', labels(ind));
end
fprintf(1, 'LinReg: R2=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.r2_LinReg), std(res.r2_LinReg));


% Try classifying using LASSO linear regression and the AUC
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [w, b] = ...
    DGOA_LinearRegression(dgoaNorm(:,~ind), labels(~ind), 'lasso');
  yTestProb = w' * dgoaNorm(:,ind) + b;
  y = labels(ind);
  res.r2_Lasso(k) = 1 - sum((yTestProb - y).^2) / sum((y - mean(y)).^2);
  
  % Evaluate the ROC and AUC
  [res.tpr_Lasso{k}, res.fpr_Lasso{k}, res.auc_Lasso(k)] = ...
    EvaluateROC(yTestProb', labels(ind));
end
fprintf(1, 'LASSO: R2=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.r2_Lasso), std(res.r2_Lasso));


% Try classifying using Gaussian SVR and the AUC
paramSVR = struct('type', 3, 'kernel', 2, 'epsilon', 0.45);
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.r2_SVRgauss(k), dummy1, yTestProb, dummy3, modelSVR] = ...
    SVM_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 'testDuval', paramSVR);
  res.nSV_SVRgauss(k) = modelSVR.totalSV;

  % Evaluate the ROC and AUC
  [res.tpr_SVRgauss{k}, res.fpr_SVRgauss{k}, res.auc_SVRgauss(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'SVRgauss: R2=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.r2_SVRgauss), std(res.r2_SVRgauss));


% Try classifying using quadratic (polynomial) SVR and the AUC
paramSVR = struct('type', 3, 'kernel', 1, 'epsilon', 0.45);
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.r2_SVRquad(k), dummy1, yTestProb, dummy3, modelSVR] = ...
    SVM_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 'testDuval', paramSVR);
  res.nSV_SVRquad(k) = modelSVR.totalSV;

  % Evaluate the ROC and AUC
  [res.tpr_SVRquad{k}, res.fpr_SVRquad{k}, res.auc_SVRquad(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'SVRquad: R2=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.r2_SVRquad), std(res.r2_SVRquad));


% Try classifying using KNN
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_KNN(k), dumy1, dumy2, dumy3, net, res.nHidden_KNN(k)] = ...
    KNN_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind));
end
fprintf(1, 'KNN: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_KNN), std(res.accuracy_KNN));


% Try classifying using C4.5 trees at 99% correct classifications
p = 1;
for k = 1:n_val
  ok_c45 = 0;
  while ~ok_c45
    ind = (rand(1, n_samples) > thres_xval);
    try
      yTestProb = ...
        C4_5(dgoaNorm(:,~ind), labels(~ind), dgoaNorm(:,ind), p);
      res.accuracy_C45(k) = ...
        sum(labels(ind) == yTestProb) / sum(ind) * 100;
      ok_c45 = 1;
    catch
      fprintf(1, 'Need to restart C4.5 tree #%2d/%2d\n', k, n_val);
    end
  end
end
fprintf(1, 'C4.5: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_C45), std(res.accuracy_C45));


% Try classifying using single hidden layer MLP
paramsNN.algorithm = 'quasinew';
paramsNN.output = 'logistic';
paramsNN.n_epochs = 100;
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_MLP(k), dummy1, yTestProb, dummy3, ...
    net, res.nHidden_MLP(k)] = ...
    NN_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), paramsNN);

  % Evaluate the ROC and AUC
  [res.tpr_MLP{k}, res.fpr_MLP{k}, res.auc_MLP(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'MLP: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_MLP), std(res.accuracy_MLP));


% Try classifying using linear SVM
paramSVM = struct('type', 0, 'kernel', 0);
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_SVMlin(k), dummy1, dummy2, dummy3, modelSVM, ...
    dummy4, dummy5, yTestProb] = ...
    SVM_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 'testDuval', paramSVM);
  res.nSV_SVMlin(k) = modelSVM.totalSV;

  % Evaluate the ROC and AUC
  [res.tpr_SVMlin{k}, res.fpr_SVMlin{k}, res.auc_SVMlin(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'SVMlin: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_SVMlin), std(res.accuracy_SVMlin));


% Try classifying using quadratic (polynomial) SVM
paramSVM = struct('type', 0, 'kernel', 1);
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_SVMquad(k), dummy1, dummy2, dummy3, modelSVM, ...
    dummy4, dummy5, yTestProb] = ...
    SVM_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 'testDuval', paramSVM);
  res.nSV_SVMquad(k) = modelSVM.totalSV;

  % Evaluate the ROC and AUC
  [res.tpr_SVMquad{k}, res.fpr_SVMquad{k}, res.auc_SVMquad(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'SVMquad: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_SVMquad), std(res.accuracy_SVMquad));


% Try classifying using Gaussian SVM, use 10% held-out data
paramSVM = struct('type', 0, 'kernel', 2);
for k = 1:n_val
  ind = (rand(1, n_samples) > thres_xval);
  [res.accuracy_SVMgauss(k), dummy1, dummy2, dummy3, modelSVM, ...
    dummy4, dummy5, yTestProb] = ...
    SVM_TrainTest(dgoaNorm(:,~ind), labels(~ind), ...
    dgoaNorm(:,ind), labels(ind), 'testDuval', paramSVM);
  res.nSV_SVMgauss(k) = modelSVM.totalSV;

  % Evaluate the ROC and AUC
  [res.tpr_SVMgauss{k}, res.fpr_SVMgauss{k}, res.auc_SVMgauss(k)] = ...
    EvaluateROC(yTestProb, labels(ind));
end
fprintf(1, 'SVMgauss: accuracy=%.2f%% (+/-%.2f%%)\n', ...
  mean(res.accuracy_SVMgauss), std(res.accuracy_SVMgauss));


% -------------------------------------------------------------------------
function [tpr, fpr, auc] = EvaluateROC(yTestProbMat, labels)

% Retrieve bad transformers in the order
yTestProb = yTestProbMat(:, 1);
if (numel(yTestProbMat) > length(yTestProb))
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

