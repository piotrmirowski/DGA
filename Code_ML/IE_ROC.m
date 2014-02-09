% IE_ROC  Compute the ROC curve and its convex hull
%
% Syntax:
%   [tpr, fpr, isRelevant, tp, fp, indConvHull] = ...
%     IE_ROC(refClassOrdered, class_query, n_relevant, get_convex_hull)
% Inputs:
%   refClassOrdered: matrix of size <K> x <T> of <K>-class membership
%                    of <T> retrieved documents, sorted by the order
%                    of retrieval.
%   class_query:     vector of size <K> x 1 of <K>-class membership of
%                    the query document
%   n_relevant:      total number of relevant documents (note that not all
%                    relevant documents are necessarily retrieved in
%                    <refClassOrdered>.
%   get_convex_hull: (optional) equal to 1 if convex hull
% Outputs:
%   tpr:         true positive rate, vector of length <n_retrieved>
%   fpr:         false positive rate, vector of length <n_retrieved>
%   isRelevant:  vector of <n_retrieved> booleans for each retrieved doc
%   tp:          count of true positives at each retrieved document,
%                vector of length <n_retrieved>
%   fp:          count of false positives at each retrieved document,
%                vector of length <n_retrieved>
%   indConvHull: indexes (ranks of) retrieved documents that lie on the
%                convex hull of the ROC curve. Note that it uses
%                Matlab's <convhull> function.
%
% Reference:
% Jesse Davis, Mark Goadrich, "The Relantionship Between Precision-Recall
% and ROC Curves", ICML 2006

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

function [tpr, fpr, isRelevant, tp, fp, indConvHull] = ...
  IE_ROC(refClassOrdered, class_query, n_relevant, get_convex_hull)

% Evaluate the retrieval (how many relevant documents retrieved)
isRelevant = IE_CompareClasses(refClassOrdered, class_query);
relevantRetrieved = cumsum(isRelevant);
n_retrieved = size(refClassOrdered, 2);
rankRetrieved = 1:n_retrieved;

% True/false positives/negatives
tp = relevantRetrieved;
fp = rankRetrieved - relevantRetrieved;
fn = n_relevant - relevantRetrieved;
tn = n_retrieved - tp - fp - fn;

% True/false positive rates
tpr = tp ./ (tp + fn);
tpr(isnan(tpr)) = 0;
fpr = fp ./ (fp + tn);

if ((nargin > 3) && (get_convex_hull))
  % Convex hull on the true positive rate
  try
    indConvHull = convhull(fpr, tpr);

    % Handle point "below" the ROC curve
    indSigns = diff(indConvHull);
    indNeg = find(indSigns < 0);
    indPos = find(indSigns > 0);
    n_neg = length(indNeg);
    n_pos = length(indPos);
    if ((n_pos > n_neg) && (n_neg > 1))
      indConvHull(indNeg(2:end)) = indConvHull(indNeg(1));
    end
    if ((n_pos < n_neg) && (n_pos > 1))
      indConvHull(indPos(2:end)) = indConvHull(indPos(1));
    end
    indConvHull = unique(indConvHull);
  catch
    indConvHull = rankRetrieved;
  end
else
  % No convex hull, return all points
  indConvHull = rankRetrieved;
end
