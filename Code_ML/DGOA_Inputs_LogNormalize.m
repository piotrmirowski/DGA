% DGOA_Inputs_LogNormalize  Log-normalize the DGOA input data
%
% Syntax:
%   [y, labels] = DGOA_Inputs_LogNormalize(x, norm_type, labels)
% Inputs:
%   x:         matrix of DGOA data, expressed in ppm
%   norm_type: string indicating the normalization type, e.g. 'log10'
%   labels:    cell array of labels for each channel, before normalization
% Outputs:
%   y:         matrix of normalized DGOA data
%   labels:    char string or cell array of renamed labels for each channel

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

function [y, labels] = DGOA_Inputs_LogNormalize(x, norm_type, labels)

switch norm_type
  case 'log10'
    y = log10(max(x, 1));
    if ((nargout > 1) && (nargin > 2))
      if iscell(labels)
        for k = 1:length(labels)
          labels{k} = ['log_{10}(' labels{k} ')'];
        end
      else
        labels = ['log_{10}(' labels ')'];
      end
    end      
  otherwise
    y = x;
end
