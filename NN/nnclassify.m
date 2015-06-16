function labels = nnclassify(nn, x)
    a = nnpredict(nn, x);
    [_, labels] = max(a,[],2);
end