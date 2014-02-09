% DGOA_PlotInterpolatedROC  Plot ROC curve
%
% Syntax:
%   DGOA_PlotInterpolatedROC(fpr, tpr, col)
% Inputs:
%   fpr: vector of length <N> of false positive rates
%   tpr: vector of length <N> of true positive rates
%   col: color

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
% Version 1.0, New York, 08 November 2009
% (c) 2009, Piotr Mirowski,
%     Ph.D. candidate at the Courant Institute of Mathematical Sciences
%     Computer Science Department
%     New York University
%     719 Broadway, 12th Floor, New York, NY 10003, USA.
%     email: mirowski [AT] cs [DOT] nyu [DOT] edu

function DGOA_PlotInterpolatedROC(fpr, tpr, col)

[fpr, ind] = sort(fpr);
tpr = tpr(ind);
[fpr, ind] = unique(fpr);
tpr = tpr(ind);

x = [0:0.01:1];
y = interp1(fpr, tpr, x, 'cubic');

for k = 2:length(y)
  y(k) = max(y(k), y(k-1));
end

plot(x, y, '-', 'LineWidth', 2, 'Color', col);
