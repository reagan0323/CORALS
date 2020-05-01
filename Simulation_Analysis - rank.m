%% simulation studies
% This file conducts simulations for ranks selection
% Just rerun our methods under settings 1 2 and 4 with a large rank and see when
% it stops (selected rank)
% No comparison, just a proof of concept
%
%
% Gen Li, 1/7/2019
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
rec_rank=zeros(nsim,1); 
for i=1:nsim
    % generate data
    rng('shuffle')
    X=poissrnd(Lambda);%figure(3);clf;mesh(X)

    [U1,V1]=GBC(X,'poisson', struct('numBC',10,'fig',0)); % 10 is large enough candidate rank
    rec_rank(i)=rank(U1);
end

figure(1);clf;
hist(rec_rank);
sim1_rank=rec_rank;

%%  Sim2: 2D, binary overlapping
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
rec_rank=zeros(nsim,1); 
for i=1:nsim
    % generate data
    rng('shuffle')
    X=binornd(1,Pi);

    [U1,V1]=GBC(X,'bernoulli', struct('numBC',10,'fig',0)); % 10 is large enough candidate rank
    rec_rank(i)=rank(U1);
end

figure(1);clf;
hist(rec_rank);
sim2_rank=rec_rank;





%% Sim4: 3D, count, overlapping
% vs Poisson CP
rng(12102018)

% poisson tensor, rank-2 biclusters
p=[50,50,50];
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
rec_rank=zeros(nsim,1); 
for i=1:nsim
    % generate data
    rng('shuffle')
    X=poissrnd(Lambda);
    
    Va=GCC(X,'poisson', struct('numBC',10,'fig',0));
    rec_rank(i)=rank(Va{1});
end

figure(1);clf;
hist(rec_rank);
sim4_rank=rec_rank;

