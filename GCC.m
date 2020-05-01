function V=GCC(data,distr, paramstruct)
% This is the function for generalized co-clustering analysis.
% It employs a deflation algorithm and estimate one layer at a time
% (treating previous layers as offset). Loadings are sparse.
%
%
% input: 
%
%   data    p1*p2*...pK data array from some exponential family distribution
%
%   distr   string, specify the distribution, 
%            choose from 'bernoulli','poisson','normal'
%            (currently not available for other distributions)
%
%
%   paramstruct
%
%           Tol_outer   converging threshold for max principal angle
%                       between consecutive loading estimates, default = 0.1 degree
%
%           Tol_inner   converging threshold for inner iterations of IRLS
%                       for estimate of each loading, default = 0.1 degree
%
%           Niter_outer max number of outer alternations (between different loadings),
%                       default=200
%
%           Niter_inner max number of inner IRLS,  default=50
%
%           numBC       number of coclusters to identify
%                       default=1 (i.e., only find the first cocluster)
%
%           fig         scalar, 0(default) no figure output
%                       1 show tuning selection figures over iterations
%
% Output: 
%
%   V       length-K list, V{k} is a pk by numBC sparse loadings in the k
%           dimention. Not strictly orthogonal; unit norm other than 1st
%           array
%
% Originated on 8/26/2018 by Gen Li



p=size(data);
K=length(p); % number of dimensions
Index=1:K;

% initial values
Tol_outer=0.1; % overall threshold
Tol_inner=0.1; % threshold for each component 
Niter_outer=200; % max number of alternations between u and v
Niter_inner=50; % max number of IRLS iterations
numBC=1; % number of biclusters
fig=0; % whether to show figures
stophere=0; % early termination sign
if nargin > 2    %  then paramstruct is an argument
  if isfield(paramstruct,'Tol_outer') 
    Tol_outer = getfield(paramstruct,'Tol_outer') ; 
  end 
  if isfield(paramstruct,'Tol_inner') 
    Tol_inner = getfield(paramstruct,'Tol_inner') ; 
  end 
  if isfield(paramstruct,'Niter_outer') 
    Niter_outer = getfield(paramstruct,'Niter_outer') ; 
  end 
  if isfield(paramstruct,'Niter_inner') 
    Niter_inner = getfield(paramstruct,'Niter_inner') ; 
  end
  if isfield(paramstruct,'numBC')
    numBC = getfield(paramstruct,'numBC') ; 
  end 
  if isfield(paramstruct,'fig')  
      fig=getfield(paramstruct,'fig') ; 
  end 
end





% define critical functions for exponential family (entrywise calc)
if strcmpi(distr,'bernoulli') 
    fcn_b=@(theta)log(1+exp(theta));
    fcn_g=@(mu)log(mu./(1-mu));
    fcn_ginv=@(eta)exp(eta)./(1+exp(eta));
    fcn_db=@(theta)exp(theta)./(1+exp(theta));
    fcn_ddb=@(theta)exp(theta)./((1+exp(theta)).^2);
    fcn_dg=@(mu)1./(mu.*(1-mu));
    pseudodata=log((data*0.8+0.1)./(0.9-0.8*data)); % for initialization (0,1->0.1,0.9)
elseif strcmpi(distr,'poisson')
    fcn_b=@(theta)exp(theta);
    fcn_g=@(mu)log(mu);
    fcn_ginv=@(eta)exp(eta);
    fcn_db=@(theta)exp(theta);
    fcn_ddb=@(theta)exp(theta);
    fcn_dg=@(mu)1./mu;
    pseudodata=log(data+0.1); % for initialization (shift->0.1 to avoid 0)
elseif strcmpi(distr,'normal')
    fcn_b=@(theta)(theta.^2)/2;
    fcn_g=@(mu)(mu);
    fcn_ginv=@(eta)(eta);
    fcn_db=@(theta)(theta);
    fcn_ddb=@(theta)ones(size(theta));
    fcn_dg=@(mu)ones(size(mu));
    pseudodata=data;
end   




%%%%%% MAIN FUNCTION %%%%%%
V = {};
for k=1:K
    V{k}=zeros(p(k),numBC);
end
for r=1:numBC
    prevtensor=TensProd_GL(V,1:(r-1)); % est of biclusters from prev ranks, as offset
    % initialization
    V_curr=parafac_GL(pseudodata-prevtensor,1);
    
    
    % alternate between different directions
    diff=inf;
    niter=1;
    rec_diff=[];
    while diff>Tol_outer && niter<=Niter_outer  
        V_old=V_curr;
        
        for dir=1:K
            % unfold data and prevtensor along dir
            X= reshape(permute(data,[dir,Index(Index~=dir)]),p(dir),[]); % p(dir)*[prod(p)/p(dir)]
            offset= reshape(permute(prevtensor,[dir,Index(Index~=dir)]),p(dir),[]);
            v_fix=1; % Kron(vK,vK-1, ... v1)
            for temp=Index(Index~=dir)
                v_fix=kron(V_curr{temp},v_fix); % length-prod(p)/p(dir)
            end
            
            % est loading dir, denoted by u
            niter_u=1;
            diff_u=inf;
            u=V_curr{dir};
            while diff_u>Tol_inner && niter_u<Niter_inner % IRLS+penalty (for normal, intrinsically no iteration)
                u_old=u;           
                % calc weight matrix
                eta=v_fix*u'+offset'; % [prod(p)/p(dir)]*[p(dir)], vec()=true eta
                mu=fcn_db(eta); %[prod(p)/p(dir)]*[p(dir)]
                sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); 
                Xmat=bsxfun(@times,sw,v_fix); % W^0.5*X but condensed 
                Ymat=sw.*((eta-offset')+(X'-mu).*fcn_dg(mu)); %W^0.5*Y but condensed

                %%% insert lasso regression here (Ymat~Xmat)
                [u,~]=mylasso(Xmat,Ymat);
                % output is sparse loading u
                
                % update stopping rule
                niter_u=niter_u+1;
                diff_u=PrinAngle(u,u_old);
            end
            V_curr{dir}=u;
        
            % check if u is all zero
            if sum(u==0)==p(dir) 
                stophere=1;
                break
            end
        end
        
        % check if early stop is needed
        if stophere % exist all zero est
            break
        end
      
        
        % normalize (just the norm of loadings)
        const=1;
        for k=2:K
            const0=norm(V_curr{k},'fro');
            const=const*const0;
            V_curr{k}=V_curr{k}/const0;
        end
        V_curr{1}=V_curr{1}*const; % put all norm to the first loading
        
    
        % stopping rule
        diff=0;
        for k=Index
            diff=diff+PrinAngle(V_curr{k},V_old{k});
        end
        rec_diff=[rec_diff,diff];
        niter=niter+1;

        
        % plot
        if fig
            figure(101);clf;
            plot(1:(niter-1),rec_diff);
            title('Sum of Angle diff for Loadings')
        end

    end
    
    % check if r'th bicluster exists
    if stophere 
        disp(['The ',num2str(r),'th cocluster does NOT exist. Return with ',num2str(r-1) ,' clusters...'])
        break
    end
    
    % check if r'th bicluster alternation converges
    if niter>Niter_outer
       disp([num2str(r),'th cocluster: NOT converge after ',num2str(Niter_outer),' iterations!']);      
    else
       disp([num2str(r),'th cocluster: converge after ',num2str(niter-1),' iterations.']);      
    end


    for k=Index
        V{k}(:,r)=V_curr{k};
    end
    

end
  




end








function angle = PrinAngle(V1,V2,paramstruct)
% if ind=1 (default), calc max principal angle
% if ind=2, Calculates All principal angles between column space of V1 and V2
ind=1;
if nargin > 2 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'ind');
      ind=getfield(paramstruct,'ind');
  end;
end;

[p1,r1]=size(V1);
[p2,r2]=size(V2);
if (p1~=p2) 
    error('Input must be matched')
end;

[V1,~,~]=svd(V1,'econ');
[V2,~,~]=svd(V2,'econ');
if ind==1
    angle=180/pi*acos(min(svd(V1'*V2)));
elseif ind==2
    angle=180/pi*acos(svd(V1'*V2));
end

end




function [beta,lambda] = mylasso(Xmat,Ymat)
% This function provides explicit solution to lasso with orthogonal design
% 1/2(y-X*beta)^2+lambda|beta|, where X'X is diagonal (but may not be I)
% Moreover, our focused X is block diagonal (each block being a column in Xmat)
% correspondingly, we condense y in a matrix form Ymat

% OLS
XtX=diag(Xmat'*Xmat); % X'*X, diagonal matrix
XtY=diag(Xmat'*Ymat); % X'*y
beta_OLS=(1./XtX).*XtY; % 
nn=size(Xmat,1)*size(Xmat,2); % sample size in lasso regression

lambda_cand=((0:100)/100*max(abs(beta_OLS).*XtX)); 

BIC_rec=zeros(1,101);
for i=1:101
    lambda=lambda_cand(i);
    % lasso solution
    beta=sign(beta_OLS).*max((abs(beta_OLS)-lambda./XtX),0);

    % BIC
    sigma2=norm(Ymat-bsxfun(@times,Xmat,beta'),'fro')^2/nn; % SSE/sample size
    BIC_rec(i)=log(sigma2)+sum(beta~=0)*log(nn)/nn;
end
[~,index]=min(BIC_rec);
lambda=lambda_cand(index);
beta=sign(beta_OLS).*max((abs(beta_OLS)-lambda./XtX),0);

% figure();clf;
% plot(lambda_cand,BIC_rec)

end
