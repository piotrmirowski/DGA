% DGOA_PlotDuval  Plot DGOA variables as Duval plot
%
% Syntax:
%   DGOA_PlotDuval(x, y, labels, labels_y, labels_type)
% Inputs:
%   x:        matrix of size 3 x <N> of DG variables and <N> samples,
%             where the variables are percentages of hydrocarbures
%   y:        vector of length <N> target values for <x>
%   labels:   cell array of char string with names of the variables
%   labels_y: cell array of char string with names of the classes in <y>
%   labels_type: label

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

function DGOA_PlotDuval(x, y, labels, labels_y, labels_type)

% Get the 3 Duval triangle components
if ((length(labels) ~= 3) || (size(x, 1) ~= 3))
  error('You need to select 3 dimensions in <x> and <labels> first');
end
label_1 = 'Duval C_2H_2';
label_2 = 'Duval CH_4';
label_3 = 'Duval C_2H_4';
for k = 1:3
  if isequal(labels{k}, label_1)
    x1 = x(k, :);
  elseif isequal(labels{k}, label_2)
    x2 = x(k, :);
  elseif isequal(labels{k}, label_3)
    x3 = x(k, :);
  else
    error('Unrecognized label %s', labels{k});
  end
end

% Plot the values
figure;
hold on;
nMeasures = size(x, 2);
if ((nargin == 5) && isequal(y, round(y)))
  symb = '.ox+*sdv^<>ph';
  col = 'rgbcmyk';
  for c = 1:max(y)
    ind = (y == c);
    [u, v] = ComputeDuval(x1(ind), x2(ind), x3(ind));
    plot(u, v, [col(mod(c-1,7)+1) symb(mod(c-1,13)+1)]);
  end
  legend(labels_y);
  title(['DGOA Duval triangle per category ' labels_type]);
else
  for k = 1:nMeasures
    col = [1-y(k), y(k), 0];
    [u, v] = ComputeDuval(x1(k), x2(k), x3(k));
    plot(u, v, '.', 'Color', col);
  end
  title('DGOA Duval triangle, red: y=0 to green: y=1');
end

% Plot the triangle
set(gcf, 'Color', [1 1 1]);
plot([0, 50, 100, 0], [0, 100 * sin(pi/3), 0, 0], 'k-');
for k = 10:10:90
  plot(k/2 + [0 5], k * sin(pi/3) * [1 1], 'k-');
  text(k/2 - 5, k * sin(pi/3), num2str(k));
  plot(50 + k/2 - [0 2.5], (100 - k - [0 5]) * sin(pi/3), 'k-');
  text(50 + k/2 + 2, (100 - k) * sin(pi/3), num2str(k));
  plot(100 - k - [0 2.5], [0 5] * sin(pi/3), 'k-');
  text(100 - k, -3, num2str(k));
end
text(50, -8, label_1);
text(10, 50, label_2);
text(80, 50, label_3);
set(gca, 'XLim', [0 100], 'YLim', [0 90], 'XTick', [], 'YTick', []);

% -------------------------------------------------------------------------
function [u, v] = ComputeDuval(x1, x2, x3)

if any((abs(x1 + x2 + x3 - 1) < 2e-2))
  x1 = x1 * 100;
  x2 = x2 * 100;
  x3 = x3 * 100;
end
if any(~((abs(x1 + x2 + x3 - 100) < 2e-2) | (x1 + x2 + x3 == 0)))
  error('x1=%g, x2=%g, x3=%g sum up to %g', x1, x2, x3, x1+x2+x3);
end
if (any(x1 > 100) || any(x1 < 0) || any(x2 > 100) || any(x2 < 0) || ...
    any(x3 > 100) || any (x3 < 0))
  error('x1=%g, x2=%g, x3=%g are not between 0 and 100', x1, x2, x3);
end
u = 100 - x1 - x2/2;
v = x2 * sin(pi/3);
