% ML_MeasurePerformance  R2 correlation or accuracy for prediction
%
% Syntax:
%   perf = ML_MeasurePerformance(y, yBar, regression)
% Inputs:
%   y:          matrix of size <dim_y> x <n> of target outputs
%   yBar:       matrix of size <dim_y> x <n> of predicted outputs
%   regression: boolean indicator: are we doing linear regression?
% Outputs:
%   perf:       performance measure: R2 correlation or accuracy
%
% Reference:
%   Nadaraya, E. A. (1964). "On Estimating Regression".
%   Theory of Probability and its Applications 9 (1): 141?142

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
% Version 1.0, New York, 18 September 2010
% (c) 2010, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function perf = ML_MeasurePerformance(y, yBar, regression)

if (regression)
  % R2 correlation
  dY2 = sum((y - yBar).^2, 1);
  mu = mean(y, 2);
  dYmu = sum((y - repmat(mu, 1, size(y, 2))).^2);
  perf = 1 - sum(dY2) / sum(dYmu);
else
  % Accuracy (in percentage)
  perf = 100 * mean(sum(round(yBar) == y, 2) / size(y, 2));
end
