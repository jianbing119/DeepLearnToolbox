function labels = nnclassify(nn, x)
    a = nnpredict(nn, x);
    [~, labels] = max(a,[],2);
end