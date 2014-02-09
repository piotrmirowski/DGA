% Script to plot the results of Duval ML analysis.
% One might need to modify <file_root>.

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

% TO CUSTOMIZE --------------------
file_root = 'Duval_Statistics_bis';
% ---------------------------------

fracs = [0.3:0.1:0.7];
for i = 1:5
  for j = 1:5
    f = fracs(j); 
    res{i, j} = ConEd_Duval_SaveTxt(sprintf('%s_%d_%g', file_root, i, f));
  end
end

% ---------------------
% Regression algorithms
% ---------------------

regressNames = ...
  {'LinReg', 'Lasso', 'NNgauss', 'SVRquad', 'SVRgauss', 'WKR', 'LLR'};
n_regress = length(regressNames);
cols = {'b-', 'c-', 'g-', 'm-', 'r-', 'k-', 'kx-'};
colsThin = {'b--', 'c--', 'g--', 'm--', 'r--', 'k--', 'k--'};

% Collect and plot the R2 correlation
figure;
hold on;
for k = 1:n_regress
  for j = 1:5
    vals = [];
    for i = 1:5
      eval(sprintf('vals = [vals res{%d,%d}.r2_%s];', ...
        i, j, regressNames{k}));
    end
    meanRegress{k}(j) = mean(vals);
    stdRegress{k}(j) = std(vals);
  end
  plot(fracs, meanRegress{k}, cols{k}, 'LineWidth', 2);
end
% Add the error bars
for k = 1:n_regress
  plot(fracs, meanRegress{k}+stdRegress{k}, colsThin{k}, 'LineWidth', 0.5);
  plot(fracs, meanRegress{k}-stdRegress{k}, colsThin{k}, 'LineWidth', 0.5);
end
legend(regressNames);
ylabel('R^2 correlation');
xlabel('Fraction of "negative" datapoints (not faulty transformers)');
title('Comparative predictive performance, averaged over 50 runs');
grid on;


% Collect and plot the AUC
figure;
hold on;
for k = 1:n_regress
  for j = 1:5
    vals = [];
    for i = 1:5
      eval(sprintf('vals = [vals res{%d,%d}.auc_%s];', ...
        i, j, regressNames{k}));
    end
    meanRegress{k}(j) = mean(vals);
    stdRegress{k}(j) = std(vals);
  end
  plot(fracs, meanRegress{k}, cols{k}, 'LineWidth', 2);
end
% Add the error bars
for k = 1:n_regress
  plot(fracs, meanRegress{k}+stdRegress{k}, colsThin{k}, 'LineWidth', 0.5);
  plot(fracs, meanRegress{k}-stdRegress{k}, colsThin{k}, 'LineWidth', 0.5);
end
legend(regressNames);
ylabel('Area under the ROC curve');
xlabel('Fraction of "negative" datapoints (not faulty transformers)');
title('Comparative predictive performance, averaged over 50 runs');
grid on;
set(gca, 'YLim', [0.8 1]);


% -------------------------
% Classification algorithms
% -------------------------

classNames = ...
  {'KNN', 'C45', 'SVMlin', 'SVMquad', 'SVMgauss', 'MLP', 'NNlog'};
n_class = length(classNames);
cols = {'b-', 'c-', 'g-', 'm-', 'r-', 'k-', 'kx-'};
colsThin = {'b--', 'c--', 'g--', 'm--', 'r--', 'k--', 'k--'};

% Collect and plot the R2 correlation
figure;
hold on;
for k = 1:n_class
  for j = 1:5
    vals = [];
    for i = 1:5
      eval(sprintf('vals = [vals res{%d,%d}.accuracy_%s];', ...
        i, j, classNames{k}));
    end
    meanClass{k}(j) = mean(vals);
    stdClass{k}(j) = std(vals);
  end
  plot(fracs, meanClass{k}, cols{k}, 'LineWidth', 2);
end
% Add the error bars
for k = 1:n_regress
  plot(fracs, meanClass{k}+stdClass{k}, colsThin{k}, 'LineWidth', 0.5);
  plot(fracs, meanClass{k}-stdClass{k}, colsThin{k}, 'LineWidth', 0.5);
end
legend(classNames);
ylabel('Accuracy (%%)');
xlabel('Fraction of "negative" datapoints (not faulty transformers)');
title('Comparative predictive performance, averaged over 50 runs');
grid on;


% Collect and plot the AUC
figure;
hold on;
for k = 1:n_class
  try
    for j = 1:5
      vals = [];
      for i = 1:5
        eval(sprintf('vals = [vals res{%d,%d}.auc_%s];', ...
          i, j, classNames{k}));
      end
      meanClass{k}(j) = mean(vals);
      stdClass{k}(j) = std(vals);
    end
  catch
    meanClass{k} = zeros(1, 5);
    stdClass{k} = zeros(1, 5);
  end
  plot(fracs, meanClass{k}, cols{k}, 'LineWidth', 2);
end
% Add the error bars
for k = 1:n_regress
  plot(fracs, meanClass{k}+stdClass{k}, colsThin{k}, 'LineWidth', 0.5);
  plot(fracs, meanClass{k}-stdClass{k}, colsThin{k}, 'LineWidth', 0.5);
end
legend(classNames);
ylabel('Area under the ROC curve');
xlabel('Fraction of "negative" datapoints (not faulty transformers)');
title('Comparative predictive performance, averaged over 50 runs');
grid on;
set(gca, 'YLim', [0.8 1]);

