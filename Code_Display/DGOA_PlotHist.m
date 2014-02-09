% DGOA_PlotHist
%
% Syntax:
%   DGOA_PlotHist(x_pos, y_pos, x_neg, labels)
% Inputs:
%   x_pos:        matrix of size <D> x <N> of <D> DG values and <N> samples
%   y_pos:        vector of length <N> target values for <x_pos>
%   x_neg:        matrix of size <D> x <M> of <D> DG values and <M> samples
%   labels:       cell array of char string with names of the gases

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

function DGOA_PlotHist(x_pos, y_pos, x_neg, labels)

nDG = size(x_pos, 1);
nP = size(y_pos, 2);
nN = size(x_neg, 2);

% Plot the distributions
nBins = 30;
for k = 1:nDG
  figure;
  hold on;

  % Plot the histogram of positives
  y = x_pos(k, :); 
  [counts, bins] = hist(y, nBins);
  h = bar(bins, counts / nP, 'r');
  set(get(h, 'Children'), 'FaceAlpha', 0.5);

  % Plot the histogram of negatives
  y = x_neg(k, :);
  [counts, bins] = hist(y, nBins);
  h = bar(bins, counts / nN, 'b');
  set(get(h, 'Children'), 'FaceAlpha', 0.5);

  % Give title and labels to the graph
  title(labels{k});
  xlabel(labels{k});
  legend({'Positives', 'Negatives'});
end
