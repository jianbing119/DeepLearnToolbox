function L = nnloss(nn, x, y)
%NNLOSS performs a feedforward pass and calculates loss
% L = nnloss(nn, x, y, batchsize) returns loss

m = size(x, 1);
batchsize = 1000;
numbatches = ceil(m / batchsize);
Ls = zeros(numbatches,1);
for i = 1 : numbatches
  batchrange = (i - 1) * batchsize + 1 : min(i * batchsize, m);
  batch_x = x(batchrange, :);
  batch_y = y(batchrange, :);
  nn.testing = 1;
  nn = nnff(nn, batch_x, batch_y);
  nn.testing = 0;
  Ls(i) = nn.L;
end
L = mean(Ls);
