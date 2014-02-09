% NN_Trainer  Train a Neural Network
%
% Syntax:
%   [model, perf_xval, perf_learn] = NN_Trainer(x, y, params)
% Inputs:
%   x:        matrix of size <dim_x> x <n1> of input features (train)
%   y:        vector of length <n1> of labels, train data
%   paramsNN: parameter struct.
%             Look at function NN_Params_Init for more explanations
% Outputs:
%   model:      structure containing the trained NN
%   perf_xval:  mean square error or accuracy on test data
%   perf_learn: mean square error or accuracy on train data
%
% References:
%   D. E. Rumelhart, G. E. Hinton, and R. J. Williams, 
%   Learning internal representations by error propagation.	
%   Cambridge, MA, USA: MIT Press, 1986, pp. 318?362.
%   Y. LeCun, L. Bottou, G. Orr, and K. Muller, "Efficient backprop",
%   in Lecture Notes in Computer Science. Berlin/Heidelberg: Springer, 1998
%   L. Bottou, "Stochastic learning",
%   in Advanced Lectures on Machine Learning, Springer-Verlag, 2004.

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

function [model, perf_xval, perf_learn] = NN_Trainer(x, y, params)

t0 = clock;

% Use default parameters
if (nargin < 3)
  params = NN_Params_Init();
end

% Check that <params> contains the necessary fields
listParams = {'dim_z', 'output', 'energy', 'perc_xval', ...
  'len_batch', 'n_epochs', 'verbose', ...
  'eta', 'lambda', 'momentum', 'eta_anneal'};
for k = 1:length(listParams)
  if ~isfield(params, listParams{k})
    error('The following parameter/field is missing in <params>: %s', ...
      listParams{k});
  end
end

% Update the dimensions of the data
params.dim_x = size(x, 1);
params.dim_y = size(y, 1);

% Initialize the model
model = NN_Model_Init(params);

% Split the dataset into learning and cross-validation sets
[datasetLearn, datasetXval] = NN_Dataset_Split(x, y, params);

% Keep memory of initial learning rates
params.eta_0 = params.eta;

% Loop on several epochs until convergence
keep_training = 1;
worse_count = 0;
measure_previous = inf;
epoch = 1;
while (keep_training)

  % Scramble the order of the training dataset
  datasetLearn = NN_Dataset_Init(datasetLearn, 1);

  % Learning passes over the training data
  t1 = clock;
  e = zeros(1, datasetLearn.n_batches);
  for k = 1:datasetLearn.n_batches
    % Get the data sample
    [x_k, y_k] = NN_Dataset_ForwardProp(datasetLearn, k);
    % Learn the current batch/sample
    [e(k), model] = NN_Learn(x_k, y_k, model, params);
  end
  
  % Evaluate the performance on the learning data
  [perf_learn, e_learn] = NN_Evaluate(datasetLearn, model, params);
  % Evaluate the performance on the x-validation data
  [perf_xval, e_xval] = NN_Evaluate(datasetXval, model, params);
  if (params.perc_xval > 0)
    measure_current = e_xval;
  else
    measure_current = e_learn;
  end
  % Trace
  if (params.verbose > 1)
    fprintf(1, 'Epoch %3d (%.2fs): ', epoch, etime(clock, t1));
    fprintf(1, 'learn (%10f, %4.3f) x-val (%10f, %4.3f)\n', ...
      e_learn, perf_learn, e_xval, perf_xval);
  end

  % Save best model
  model = NN_Model_SaveBest(model, measure_current, epoch);
  
  % Convergence (early stopping)
  epoch = epoch + 1;
  if (measure_current < measure_previous)
    worse_count = 0;
  else
    worse_count = worse_count + 1;
    if (params.verbose > 1)
      fprintf(1, 'X-val worsening for %d epochs...\n', worse_count);
    end
  end
  measure_previous = measure_current;
  keep_training = ((epoch <= params.n_epochs) & (worse_count < 3));

  % Learning rate annealing
  params.eta = params.eta * params.eta_anneal;
end

% Retrieve the best model and evaluate it
model = NN_Model_RetrieveBest(model);
perf_learn = NN_Evaluate(datasetLearn, model, params);
perf_xval = NN_Evaluate(datasetXval, model, params);
% Trace
if (params.verbose > 0)
  fprintf(1, '%d epochs: best %d (%.2fs): learn %4.3f x-val %4.3f\n', ...
    epoch, model.epoch, etime(clock, t0), perf_learn, perf_xval);
end


% -------------------------------------------------------------------------
function [perf, e, yBar] = NN_Evaluate(dataset, model, params)

% Forward propagation to get the prediction (and hidden activations)
yBar = NN_Model_ForwardProp(dataset.x, model);
if isequal(params.output, 'logistic')
  yBar = NN_Logistic(yBar);
end

% Evaluate the energy (regression or classification)
[e, perf] = NN_Energy_ForwardProp(dataset.y, yBar, params);


% -------------------------------------------------------------------------
function model = NN_Model_SaveBest(model, measure, epoch)

% Save best parameters, if this is the case
if (model.measure > measure)
  model.A_best = model.A;
  model.Abias_best = model.Abias;
  model.B_best = model.B;
  model.Bbias_best = model.Bbias;
  model.measure = measure;
  model.epoch = epoch;
end


% -------------------------------------------------------------------------
function model = NN_Model_RetrieveBest(model)

% Retrieve best parameters
model.A = model.A_best;
model.Abias = model.Abias_best;
model.B = model.B_best;
model.Bbias = model.Bbias_best;
