% DGOA_PlotDGOA3D  Plot DGOA variables in a 3D plot
%
% Syntax:
%   DGOA_PlotDGOA3D(x, y, labels, labels_y, labels_type)
% Inputs:
%   x:           matrix of size <D> x <N> of <D> DG variables and <N> samp.
%   y:           vector of length <N> target values for <x>
%   labels:      cell array of char string with names of the variables
%   labels_y:    cell array of char string with names of the classes in <y>
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

function DGOA_PlotDGOA3D(x, y, labels, labels_y, labels_type)

if ((length(labels) ~= 3) || (size(x, 1) ~= 3))
  error('You need to select 3 dimensions in <x> and <labels> first');
end

figure;
hold on;
nMeasures = size(x, 2);
if ((nargin == 5) && isequal(y, round(y)))
  symb = '.xo+*sdv^<>ph';
  col = 'brgcmyk';
  for c = 1:max(y)
    ind = (y == c);
    plot3(x(1, ind), x(2, ind), x(3, ind), ...
      [col(mod(c-1,7)+1) symb(mod(c-1,13)+1)]);
  end
  legend(labels_y);
  title(['DGOA data per category ' labels_type]);
else
  y = min(1, max(0, y));
  for k = 1:nMeasures
    col = [1-y(k), y(k), 0];
    plot3(x(1, k), x(2, k), x(3, k), '.', 'Color', col);
  end
  title('Red: y=0 (failure soon) to green: y=1 (failure later)');
end
xlabel(labels{1});
ylabel(labels{2});
zlabel(labels{3});
grid on;

% Rotate for better view
camorbit(30, -60);
