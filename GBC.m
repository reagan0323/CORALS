function [U,V]=GBC(data,distr, paramstruct)
% This is the function for generalized biclustering analysis.
% It employs a deflation algorithm and estimate one layer at a time
% (treating previous layers as offset). Loadings and scores are both sparse.
%
%
% input: 
%
%   data    n*p data matrix from some exponential family distribution
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
%           Niter_outer max number of outer alternations (between u and v),
%                       default=200
%
%           Niter_inner max number of inner IRLS,  default=50
%
%           numBC       number of biclusters to identify
%                       default=1 (i.e., only find the first bicluster)
%
%           fig         scalar, 0(default) no figure output
%                       1 show tuning selection figures over iterations
%
% Output: 
%
%   U       n*r sparse score vectors, not strictly orthogonal 
% 
%   V       p*r sparse loading vectors, not strictly orthogonal but unit-norm
%
%
% Originated on 8/20/2018 by Gen Li





[n,p]=size(data);
tdata=data';


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
U=zeros(n,numBC);
V=zeros(p,numBC);
for r=1:numBC
    prevmat=U*V'; % est of biclusters from prev ranks, as offset
    tprevmat=prevmat';
    rec_u=[];
    rec_v=[];
    rec_usp=[];
    rec_vsp=[];    
    % initialization
    [u_curr,d_curr,v_curr]=svds(pseudodata-prevmat,1);
    u_curr=u_curr*d_curr;
    
    
    % alternate between u and v
    diff=inf;
    niter=1;
    while diff>Tol_outer && niter<=Niter_outer  
        u_store=u_curr;
        v_store=v_curr; 
        
        % Est u (fix v) 
        niter_u=1; 
        diff_u=inf;
        u=u_curr;
        while diff_u>Tol_inner && niter_u<Niter_inner % IRLS+penalty (for normal, intrinsically no iteration)
            u_old=u;           
            % calc weight matrix
            eta=v_curr*u'+tprevmat; % p*n, vec()=true eta
            mu=fcn_db(eta); %p*n
            sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); % p*n, sqrt of diagonal of weight matrix for IRLS
            Xmat=bsxfun(@times,sw,v_curr); % W^0.5*X but condensed as p*n
            Ymat=sw.*((eta-tprevmat)+(tdata-mu).*fcn_dg(mu)); %W^0.5*Y but condensed as p*n

            %%% insert lasso regression here (Ymat~Xmat)
            [u,~]=mylasso(Xmat,Ymat);
            % output is sparse loading u
            
            % update stopping rule
            niter_u=niter_u+1;
            diff_u=PrinAngle(u,u_old);
        end
        u_curr=u;
        
        % check if u_curr is all zero
        if sum(u_curr==0)==n 
            stophere=1;
            break
        end
        
        
        
        % Est v (fix u) 
        niter_v=1; 
        diff_v=inf;
        v=v_curr;
        while diff_v>Tol_inner && niter_v<Niter_inner % IRLS+penalty
            v_old=v;     
            % calc weight matrix
            eta=u_curr*v'+prevmat; % n*p
            mu=fcn_db(eta); %n*p
            sw=1./sqrt(fcn_ddb(eta).*((fcn_dg(mu)).^2)); % n*p, diagonal of weight matrix for IRLS
            Xmat=bsxfun(@times,sw,u_curr); % n*p
            Ymat=sw.*((eta-prevmat)+(data-mu).*fcn_dg(mu)); %n*p

            %%% insert lasso regression here (Ymat~Xmat) with BIC 
            [v,~]=mylasso(Xmat,Ymat);
            % output is sparse loading v
            
            % update stopping rule
            niter_v=niter_v+1;
            diff_v=PrinAngle(v,v_old);
        end
        v_curr=v;
        
        % check if v_curr is all zero
        if sum(v_curr==0)==p 
            stophere=1;
            break
        end
        
        
        % normalize (just the norm of v)
        temp=norm(v_curr,'fro');
        v_curr=v_curr/temp;
        u_curr=u_curr*temp;
        
        
        % plot
        rec_v=[rec_v,PrinAngle(v_store,v_curr)];
        rec_u=[rec_u,PrinAngle(u_store,u_curr)];
        rec_vsp=[rec_vsp,sum(v_curr==0)];
        rec_usp=[rec_usp,sum(u_curr==0)];
        if fig
            figure(101);clf;
            subplot(2,2,1); 
            plot(1:niter,rec_v);
            title('Angle diff for v')
            subplot(2,2,2);
            plot(1:niter,rec_u);
            title('Angle diff for u')
            subplot(2,2,3); 
            plot(1:niter,rec_vsp);
            title('sparsity for v')
            subplot(2,2,4); 
            plot(1:niter,rec_usp);
            title('sparsity for u')
        end
        drawnow
        
        
        % stopping rule
        diff=PrinAngle(v_store,v_curr); % just compare v estimate over alternations
        niter=niter+1;
    end
    
    % check if r'th bicluster exists
    if stophere 
        disp(['The ',num2str(r),'th bicluster does NOT exist. Return with ',num2str(r-1) ,' clusters...'])
        break
    end
    
    % check if r'th bicluster alternation converges
    if niter>Niter_outer
       disp([num2str(r),'th bicluster: NOT converge after ',num2str(Niter_outer),' iterations!']);      
    else
       disp([num2str(r),'th bicluster: converge after ',num2str(niter-1),' iterations.']);      
    end


    U(:,r)=u_curr;
    V(:,r)=v_curr;
    

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
end;

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

% figure(100);clf;
% plot(lambda_cand,BIC_rec)

end
