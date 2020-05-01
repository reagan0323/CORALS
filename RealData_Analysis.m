% This file conducts real data analysis using the co-clustering code GBC
% and GCC. The tuning parameter selection has been embedded.
% by Gen Li, 10/12/2018
close all
clear all
clc
%% CAL500 annotation matrix
dpth='data\';
load([dpth,'CAL500_full.mat']);
% we only care X2: annotation
colnames=X2names;
colnames{1}=['Emotion-',colnames{1}];
rownames=songnames;
size(X2)




% preprocess X2 before applying GBC
chooseanno=sum(X2,1); % max=444 (out of 502)
figure(2);clf;subplot(1,2,1);
hist(chooseanno);
title('Annotation')
keep_col=(chooseanno>=30);% only keep annotations appearing in no less than 30 songs (consis with literature)
choosesong=sum(X2,2);
figure(2);subplot(1,2,2);
hist(choosesong) % looks ok
title('Song')
keep_row=(choosesong>=0 );

newX=X2(keep_row,keep_col);
newcolnames=colnames(keep_col);
newrownames=rownames;
[newn,newp]=size(newX) % 502(songs) *99(anno)
sum(newX(:))/(newn*newp) % 21% 1
figure(1);clf;
MakeImageRB(newX);
% reorder rows/cols to check if there are clusters
row = linkage(newX, 'average', 'correlation');
col = linkage(newX', 'average', 'correlation');
prow = U_get_HC_permutation(row);
pcol = U_get_HC_permutation(col);
reorder_newX=newX(prow,pcol);
figure(1);clf;
MakeImageRB(reorder_newX);
xlabel('Annotation','fontsize',25);
ylabel('Song','fontsize',25)
title('Observed Data','fontsize',25)
set(gca,'fontsize',25)

% apply GBC 
[U,V]=GBC(newX,'bernoulli', struct('numBC',5,'fig',1));
Theta=U*V';
sum((U~=0),1)
sum((V~=0),1) 

% check results
figure(3);clf;
subplot(1,2,1);
plot(U(prow,1));
subplot(1,2,2);
plot(V(pcol,1)); % U all neg, V all pos, seems to capture the global mean only
figure(3);clf;
subplot(1,2,1);
plot(U(prow,2));
subplot(1,2,2);
plot(V(pcol,2)); % show interesting patterns, may identify two all-1 biclusters (++ and --)
figure(3);clf;
subplot(1,2,1);
plot(U(prow,3));
subplot(1,2,2);
plot(V(pcol,3)); % show interesting patterns, may identify four all-1 biclusters (++ and --)


% identified All-1 biclusters
u1=U(:,2)/norm(U(:,2),'fro');
u2=U(:,3)/norm(U(:,3),'fro');
v1=V(:,2)/norm(V(:,2),'fro');
v2=V(:,3)/norm(V(:,3),'fro');
thres_v=0.1;
thres_u=0.05; % ad hoc threshold for more sparsity
su1=(u1>thres_u)-(u1<-thres_u); % only keep +1, -1, 0
su2=(u2>thres_u)-(u2<-thres_u); 
sv1=(v1>thres_v)-(v1<-thres_v);
sv2=(v2>thres_v)-(v2<-thres_v);
[sum(su1==1),sum(sv1==1)] % all-1 bicluster1: 69*13
[sum(su1==-1),sum(sv1==-1)] % all-1 bicluster2: 71*12
[sum(su2==1),sum(sv2==1)] % all-1 bicluster3: 68*8
[sum(su2==-1),sum(sv2==-1)] % all-1 bicluster4: 59*9
bicluster12=((su1*sv1')==1);
bicluster34=((su2*sv2')==1);
allbiclusters=bicluster12+bicluster34;

% show all-1 biclusters on figure
figure(2);clf;
MakeImageRB(allbiclusters(prow,pcol));
xlabel('Annotation','fontsize',25);
ylabel('Song','fontsize',25)
title('Identified Biclusters','fontsize',25)
set(gca,'fontsize',25)

% check bicluster meaning
% bicluster X (highly interpretable)
newcolnames(sv1==1)
newrownames(su1==1)
%
newcolnames(sv1==-1)
newrownames(su1==-1)
%
newcolnames(sv2==1)
newrownames(su2==1)
%
newcolnames(sv2==-1)
newrownames(su2==-1)


%% NIPs word count data
load([dpth,'NIPs_aggregated_Data.mat']);

[n,p]=size(data); % 11463 * 29
% check
figure(1);clf;
plot(year,npaper,'ko-','markersize',5,'linewidth',2);
set(gca,'fontsize',25)
xlabel('Year','fontsize',25);
ylabel('Number of Papers','fontsize',25);


% preprocess the word counts 
rowkeep=(sum(data==0,2)==0); % only keep words that appear every year
sum(rowkeep) % 3556 out of 11463
newdata=data(rowkeep,:);
newword=word(rowkeep);

% apply GBC_v1 to X2
[U,V]=GBC(newdata,'poisson', struct('numBC',6,'fig',1));
Theta=U*V'; 
figure(2);clf;
mesh(exp(Theta))
sum((U~=0),1) 
sum((V~=0),1) 
% identify 4-1=3 biclusters, and each bicluster is indeed 2 biclusters (++ and --)

figure(1);clf;
plot(year,V(:,1),'ko-','markersize',5,'linewidth',2);
set(gca,'fontsize',25)
xlabel('Year','fontsize',25);
ylabel('Loading Value','fontsize',25);


% interpret results
u1=U(:,2)/norm(U(:,2),'fro');
u2=U(:,3)/norm(U(:,3),'fro');
u3=U(:,4)/norm(U(:,4),'fro');
v1=V(:,2)/norm(V(:,2),'fro');
v2=V(:,3)/norm(V(:,3),'fro');
v3=V(:,4)/norm(V(:,4),'fro');
thres_v=0;
thres_u=0; 
su1=(u1>thres_u)-(u1<-thres_u); % only keep +1, -1, 0
su2=(u2>thres_u)-(u2<-thres_u); 
su3=(u3>thres_u)-(u3<-thres_u); 
sv1=(v1>thres_v)-(v1<-thres_v);
sv2=(v2>thres_v)-(v2<-thres_v);
sv3=(v3>thres_v)-(v3<-thres_v);
% below are above-avg word/year
[sum(su1==1),sum(sv1==1)] % 287*19
[sum(su1==-1),sum(sv1==-1)] % 530*10
[sum(su2==1),sum(sv2==1)] % 127*18
[sum(su2==-1),sum(sv2==-1)] % 118*4
[sum(su3==1),sum(sv3==1)] % 13*21
[sum(su3==-1),sum(sv3==-1)] % 82*7

% output for excel
year(sv1==1)
newword(su1==1)


%% Sclerosis (Gaussian data)
load([dpth,'Sclerosis.mat']);

[p1,p2,p3]=size(Sclerosis); % 56 genes * 5 time pts * 12 subj
% data transformation
data=log(Sclerosis);

% apply GCC_v1 to X2 (results somewhat subj to random initial in parafac)
rng(1234)% identify 5 layers
V=GCC(data,'normal', struct('numBC',10,'fig',1)); 
sum((V{1}~=0),1) % no sparsity
sum((V{2}~=0),1)
sum((V{3}~=0),1)
% check
figure(2);clf;
plot(V{1}(:,2:5))
plot(V{2})
plot(V{3})

% the identified genes in different coclusters are similar
% we focus on the subset of 12 genes identified by V{1}(:,3)
genesig(V{1}(:,3)~=0)  % 12 genes

% look into GO for annotation and pathway enrichment analysis...


