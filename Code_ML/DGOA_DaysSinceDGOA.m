% DGOA_DaysSinceDGOA  Prepare a function of #days since DGOA measures
%
% Prepare a variable depending on the number of days since DGOA measures
%
% Syntax:
%   [y_pos, days_cdf, days] = ...
%     DGOA_DaysSinceDGOA(y_pos, label)
% Input:
%   y_pos:    vector of the number of days since DGOA measures
%   label:    name of the variable
% Outputs:
%   y_pos:    transformed variable, based on input <y_pos>, going 
%             from 0 (now) to 1 (much later)
%   days_cdf: cumulated distribution function of <y_pos>
%   days:     values at which <days_cdf> is evaluated

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

function [y_pos, days_cdf, days] = DGOA_DaysSinceDGOA(y_pos, label)

% Put nan values when no second date
y_pos(y_pos < -20000) = nan;

% If DGOA test have been done after the event, consider it same day
y_pos(y_pos < 0) = 0;

% Compute and plot the empirical CDF of the time since DGOA
[days_cdf, days] = ecdf(y_pos(~isnan(y_pos)));
figure;
stairs(days, days_cdf);
ylabel('F(x)');
xlabel(['x (' label ')']);
title(['CDF of the ' label ' (if it happened)']);
set(gca, 'XTick', [0 30 90 180 365*[1:10]], 'XLim', [0 max(days)]);
grid on;

% Transform the number of days to de-energization into numbers from 0 to 1,
% using the empirical CDF
% Convert nan values to a high number of days
y_pos(isnan(y_pos)) = 2500;
y_pos = DGOA_InverseCDF(y_pos, days, days_cdf);

% Ensure that the outputs are row vectors
y_pos = y_pos(:)';
days = days(:)';
days_cdf = days_cdf(:)';
