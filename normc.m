function out=normc(mat)
% centering each column of matrix mat
% a function in neural network toolbox
out = sqrt(bsxfun(@rdivide,mat.^2, sum(mat.^2,1))) .* sign(mat);
end