function [er, bad] = nntest(nn, x, y)
    labels = nnclassify(nn, x);
    [_, expected] = max(y,[],2);
    bad = find(labels ~= expected);
    er = numel(bad) / size(x, 1);
end
