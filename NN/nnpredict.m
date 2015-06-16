function a = nnpredict(nn, x)
    m = size(x, 1);
    d = nn.size(end);

    batchsize = 1000;
    numbatches = ceil(m / batchsize);
    a = zeros(m, d);
    for i = 1 : numbatches
      batchrange = (i - 1) * batchsize + 1 : min(i * batchsize, m);
      batch_x = x(batchrange, :);
      nn.testing = 1;
      nn = nnff(nn, batch_x, zeros(numel(batchrange), d));
      nn.testing = 0;
      a(batchrange, :) = nn.a{end};
    end
end
