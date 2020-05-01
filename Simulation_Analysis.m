%% simulation studies
% This file conducts comprehensive simulations to compare the proposed
% methods (CORALS) with competing methods
%
% We consider the following scenarios:
%   1. 2D count - overlapping
%   2. 2D binary - overlapping
%   3. 3D normal - overlapping
%   4. 3D count - overlapping
%
clear all;
close all;
clc;
rng(12102018)
addpath(genpath(['.',filesep]))
%% Sim1: 2D, count, overlapping
n=100;
p=100;

% poisson matrix, rank-2 biclusters
rng(12102018)
Utrue=[[sign(rand(50,1)-0.5).*(rand(50,1)*0.1+0.4);zeros(n-50,1)],[zeros(30,1);sign(rand(40,1)-0.5).*(rand(40,1)*0.1+0.4);zeros(n-70,1)]];
Vtrue=[[sign(rand(40,1)-0.5).*(rand(40,1)*0.1+0.4);zeros(p-40,1)],[zeros(20,1);sign(rand(40,1)-0.5).*(rand(40,1)*0.1+0.5);zeros(p-60,1)]];
Vtrue=normc(Vtrue);
Utrue=normc(Utrue)*diag([15,10]);   %*diag([20,15]);  
truebc=(Utrue*Vtrue'~=0);
Thetatrue=Utrue*Vtrue'+2;
Lambda=exp(Thetatrue); % add mean shift because Poisson data are skewed (low signal for negative Theta)


% run 100 simulations, with the same Lambda
nsim=100;
rec_nonzero=zeros(nsim,2*2); % sensitivity/specificity (larger=better)
rec_theta=zeros(nsim,2); % theta recovery (smaller=better)
rec_time=zeros(nsim,2);
for i=1:nsim
    % generate data
    rng('shuffle')
    X=poissrnd(Lambda);figure(3);clf;mesh(X)
    
    
    % CORALS
    tic;
    [U1,V1]=GBC(X,'poisson', struct('numBC',3,'fig',0));
    Theta1=U1*V1'; % in the natural parameter space
    estbc=(U1(:,2:3)*V1(:,2:3)'~=0);
    sen1=sum(sum((estbc&truebc)))/sum(truebc(:));
    spc1=sum(sum((estbc==0)&(truebc==0)))/sum(1-truebc(:));
    T1=toc;
    
    % SSVD (logX)
    tic;
    logX=log(max(X,0.5));
    [U2,V2]=GBC(logX,'normal', struct('numBC',3,'fig',0)); % in natural param space
    Theta2=U2*V2';
    estbc=(U2(:,2:3)*V2(:,2:3)'~=0);
    sen2=sum(sum((estbc&truebc)))/sum(truebc(:));
    spc2=sum(sum((estbc==0)&(truebc==0)))/sum(1-truebc(:));
    T2=toc;
    

    
    % record results
    rec_nonzero(i,:)=[sen1,spc1,sen2,spc2];
    rec_theta(i,:)=[norm(Theta1-Thetatrue,'fro'),norm(Theta2-Thetatrue,'fro')];
    rec_time(i,:)=[T1,T2];
end
[mean(rec_theta);std(rec_theta)]
[mean(rec_nonzero);std(rec_nonzero)]
[mean(rec_time);std(rec_time)]



%%  Sim2: 2D, binary, overlapping
n=100;
p=100;

% binary matrix, rank-2 biclusters
rng(12102018)
Utrue=[[rand(50,1)-0.5;zeros(n-50,1)],[zeros(30,1);rand(40,1)-0.5;zeros(n-70,1)]];
Vtrue=[[rand(40,1)-0.5;zeros(p-40,1)],[zeros(20,1);rand(40,1)-0.5;zeros(p-60,1)]];
Vtrue=normc(Vtrue);
Utrue=normc(Utrue)*diag([100,80]);
truebc=(Utrue*Vtrue'~=0);
Thetatrue=Utrue*Vtrue';
Pi=exp(Thetatrue)./(1+exp(Thetatrue)); % add mean shift because Poisson data are skewed (low signal for negative Theta)



% run 100 simulations, with the same Lambda
nsim=100;
rec_nonzero=zeros(nsim,2); % sensitivity/specificity (larger=better)
rec_theta=zeros(nsim,1); % theta recovery (smaller=better)
rec_time=zeros(nsim,1);
for i=1:nsim
    % generate data
    rng('shuffle')
    X=binornd(1,Pi);
    
    % CORALS
    tic;
    [U1,V1]=GBC(X,'bernoulli', struct('numBC',2,'fig',0));
    Theta1=U1*V1'; % in the natural parameter space
    estbc=(U1*V1'~=0);
    sen1=sum(sum((estbc&truebc)))/sum(truebc(:));
    spc1=sum(sum((estbc==0)&(truebc==0)))/sum(1-truebc(:));
    T1=toc;
       
    % record results
    rec_nonzero(i,:)=[sen1,spc1];
    rec_theta(i,:)=[norm(Theta1-Thetatrue,'fro')];
    rec_time(i,:)=[T1];
end

rec_nonzero
rec_theta
[mean(rec_theta);std(rec_theta)]
[mean(rec_nonzero);std(rec_nonzero)]
[mean(rec_time);std(rec_time)]



%% Sim3**: 3D, normal, overlapping (vary SNR to generate a figure)
% vs CP
rec_SNR=[];
SNR_cand=10.^[-1:0.1:0];
rng(20190312)
p=[50,50,50];
V1true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(1)-20,1)],...
    [zeros(10,1);sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(1)-30,1)]];
V2true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(2)-20,1)],...
    [zeros(10,1);sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(2)-30,1)]];
V3true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(3)-20,1)],...
    [zeros(10,1);sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(3)-30,1)]];
V1true=normc(V1true)* diag([80,50]);
V2true=normc(V2true);
V3true=normc(V3true);
Thetatrue=TensProd_GL({V1true,V2true,V3true},1:2);
truecc=Thetatrue~=0;


for ind=1:length(SNR_cand)
    SNR=SNR_cand(ind);
    

    sigma=sqrt(var(Thetatrue(:))/SNR);

    disp(['preset SNR: ',num2str(SNR)]);

    % run 100 simulations, with the same Theta
    nsim=100;
    rec_nonzero=zeros(nsim,2); % sensitivity/specificity (larger=better)
    rec_theta=zeros(nsim,2); % theta recovery (smaller=better)
    rec_time=zeros(nsim,2);
    X_forR=zeros(p(1),p(2),p(3),nsim);

    for i=1:nsim
        % generate data
        rng('shuffle');
        X=normrnd(Thetatrue,sigma);

        % CORALS
        tic;
        Va=GCC(X,'normal', struct('numBC',2,'fig',0));
        Theta1=TensProd_GL(Va,1:2);
        estcc=(Theta1~=0);
        temp1=estcc&truecc;
        temp2=(estcc==0)&(truecc==0);
        sen1=sum(temp1(:))/sum(truecc(:));
        spc1=sum(temp2(:))/sum(1-truecc(:));
        T1=toc;

        % CP
        tic;
        [Vb,temp]=parafac_GL(X,2);
        Theta2=TensProd_GL(Vb,1:2);
        T2=toc;

        % record results
        rec_nonzero(i,:)=[sen1,spc1];
        rec_theta(i,:)=[norm(Theta1(:)-Thetatrue(:),'fro'),norm(Theta2(:)-Thetatrue(:),'fro')];
        rec_time(i,:)=[T1,T2];
        X_forR(:,:,:,i)=X;
    end 
    rec_SNR=[rec_SNR,rec_theta]; % every 2 columns form a group
end;




%% Sim4: 3D, count, overlapping
% vs Poisson CP
% poisson tensor, rank-2 biclusters
p=[50,50,50];
rng(12102018)
V1true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(1)-20,1)],...
    [zeros(10,1);sign(rand(30,1)-0.5).*(rand(30,1)*0.1+0.4);zeros(p(1)-40,1)]];
V2true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(2)-20,1)],...
    [zeros(10,1);sign(rand(30,1)-0.5).*(rand(30,1)*0.1+0.4);zeros(p(2)-40,1)]];
V3true=[[sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(3)-20,1)],...
    [zeros(20,1);sign(rand(20,1)-0.5).*(rand(20,1)*0.1+0.4);zeros(p(3)-40,1)]];
V1true=normc(V1true)*diag([50,40]);
V2true=normc(V2true);
V3true=normc(V3true);
Thetatrue=TensProd_GL({V1true,V2true,V3true},1:2)+2; % add mean shift because Poisson data are skewed (low signal for negative Theta)
truecc=(Thetatrue~=2);
Lambda=exp(Thetatrue); 

% run 100 simulations, with the same Theta
nsim=100;
rec_nonzero=zeros(nsim,2); % sensitivity/specificity (larger=better)
rec_theta=zeros(nsim,2); % theta recovery (smaller=better)
rec_time=zeros(nsim,2);
for i=1:nsim
    % generate data
    rng('shuffle')
    X=poissrnd(Lambda);figure(1);hist(X(:));
           
    % CORALS
    tic;
    Va=GCC(X,'poisson', struct('numBC',4,'fig',0));
    Theta1cc=TensProd_GL(Va,2:3); % only 2:3 has sparsity; layer1 capture mean shift
    estcc=(Theta1cc~=0);
    temp1=estcc&truecc;
    temp2=(estcc==0)&(truecc==0);
    sen1=sum(temp1(:))/sum(truecc(:));
    spc1=sum(temp2(:))/sum(1-truecc(:));
    Theta1=TensProd_GL(Va);
    T1=toc;
    
    % logX CP
    tic;
    logX=log(max(X,0.5));
    [Vb,temp]=parafac_GL(logX,3);
    Theta2=TensProd_GL(Vb);
    T2=toc;
    
    
    
    % record results
    rec_nonzero(i,:)=[sen1,spc1];
    rec_theta(i,:)=[norm(Theta1(:)-Thetatrue(:),'fro'),norm(Theta2(:)-Thetatrue(:),'fro')];
    rec_time(i,:)=[T1,T2];
end


boxplot(rec_theta)
[mean(rec_theta);std(rec_theta)]
[mean(rec_nonzero);std(rec_nonzero)]
[mean(rec_time);std(rec_time)]

