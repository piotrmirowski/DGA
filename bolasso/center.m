function X = center(X)
% CENTER  Center the observations of a data matrix.
%    X = CENTER(X) centers the observations of an nxp data matrix where n
%    is the number of observations and p is the number of variables.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk

[n p] = size(X);
X = X - ones(n,1)*mean(X);
