clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));
path5=[ '../LaplaceManifold/Sleep/'];
addpath(genpath(path5));

Tmax=274;  %%274

NSUB=971;
N=62;

LATDIM=7;
Isubdiag = find(tril(ones(LATDIM),-1));

indexregion=[1:31 50:80];

TR=0.72;  % Repetition Time (seconds)
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

%% Resting

load('hcp1003ordered_REST1_LR_dbs80.mat');

for sub=1:NSUB
    sub
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    Tm=size(ts,2);

    %% Diff Gauss

    Kmatrix=zeros(Tm,Tm);
    epsilon=400;

    for i=1:Tm
        for j=1:Tm
            dij2=sum((ts(:,i)-ts(:,j)).^2);
            Kmatrix(i,j)=exp(-dij2/epsilon);
        end
    end

    Dmatrix=diag(sum(Kmatrix,2));
    Pmatrix=inv(Dmatrix)*Kmatrix;
    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));

    zPhi=zscore(Phi);
    Covar=corrcoef(Phi);
    EntropyC_rest(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

    for t=1:Tm
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=dist(Edges);
    MetaC_rest(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;

    %% QDM
    epsilon=300;
    Thorizont=3;

    Kmatrix=zeros(Tm,Tm);

    for i=1:Tm
        for j=1:Tm
            dij2=sum((ts(:,i)-ts(:,j)).^2);
            Kmatrix(i,j)=exp(complex(0,1)*dij2/epsilon);
        end
    end

    Ktr_t=Kmatrix^Thorizont;
    Ptr_t=abs(Ktr_t).^2;

    Dmatrix=diag(sum(Ptr_t,2));
    Pmatrix=inv(Dmatrix)*Ptr_t;

    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

    zPhi=zscore(Phi);
    Covar=corrcoef(Phi);

    EntropyQ_rest(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

    for t=1:Tm
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=dist(Edges);
    MetaQ_rest(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
end

%% Social

load('hcp1003ordered_SOCIAL_LR_dbs80.mat');
clear signal_filt Edges;

for sub=1:NSUB
    sub
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    Tm=size(ts,2);

    %% Diff Gauss

    Kmatrix=zeros(Tm,Tm);
    epsilon=400;

    for i=1:Tm
        for j=1:Tm
            dij2=sum((ts(:,i)-ts(:,j)).^2);
            Kmatrix(i,j)=exp(-dij2/epsilon);
        end
    end

    Dmatrix=diag(sum(Kmatrix,2));
    Pmatrix=inv(Dmatrix)*Kmatrix;
    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));

    zPhi=zscore(Phi);
    Covar=corrcoef(Phi);
    EntropyC_social(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

    for t=1:Tm
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=dist(Edges);
    MetaC_social(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;

    %% QDM
    epsilon=300;
    Thorizont=3;

    Kmatrix=zeros(Tm,Tm);

    for i=1:Tm
        for j=1:Tm
            dij2=sum((ts(:,i)-ts(:,j)).^2);
            Kmatrix(i,j)=exp(complex(0,1)*dij2/epsilon);
        end
    end

    Ktr_t=Kmatrix^Thorizont;
    Ptr_t=abs(Ktr_t).^2;

    Dmatrix=diag(sum(Ptr_t,2));
    Pmatrix=inv(Dmatrix)*Ptr_t;

    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

    zPhi=zscore(Phi);
    Covar=corrcoef(Phi);

    EntropyQ_social(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

    for t=1:Tm
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=dist(Edges);
    MetaQ_social(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
end

%%
TL=0;
TH=100;

figure(1)
subplot(1,2,1)
[EntropyQ_social0 idx]=rmoutliers(EntropyQ_social,'percentiles',[TL TH]);
EntropyQ_rest0=EntropyQ_rest;
EntropyQ_rest0(idx)=[];
[EntropyQ_rest0 idx]=rmoutliers(EntropyQ_rest0,'percentiles',[TL TH]);
EntropyQ_social0=EntropyQ_social0;
EntropyQ_social0(idx)=[];
boxplot([EntropyQ_rest0' EntropyQ_social0']);

a=EntropyQ_rest0;
b=EntropyQ_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'signrank');
pp=min(stats.pvals)

subplot(1,2,2)
[EntropyC_social0 idx]=rmoutliers(EntropyC_social,'percentiles',[TL TH]);
EntropyC_rest0=EntropyC_rest;
EntropyC_rest0(idx)=[];
[EntropyC_rest0 idx]=rmoutliers(EntropyC_rest0,'percentiles',[TL TH]);
EntropyC_social0=EntropyC_social0;
EntropyC_social0(idx)=[];
boxplot([EntropyC_rest0' EntropyC_social0']);

a=EntropyC_rest0;
b=EntropyC_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'signrank');
pp=min(stats.pvals)


figure(2)

subplot(1,2,1)
[MetaQ_social0 idx]=rmoutliers(MetaQ_social,'percentiles',[TL TH]);
MetaQ_rest0=MetaQ_rest;
MetaQ_rest0(idx)=[];
[MetaQ_rest0 idx]=rmoutliers(MetaQ_rest0,'percentiles',[TL TH]);
MetaQ_social0=MetaQ_social0;
MetaQ_social0(idx)=[];
boxplot([MetaQ_rest0' MetaQ_social0']);

a=MetaQ_rest0;
b=MetaQ_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'signrank');
pp=min(stats.pvals)

subplot(1,2,2)
[MetaC_social0 idx]=rmoutliers(MetaC_social,'percentiles',[TL TH]);
MetaC_rest0=MetaC_rest;
MetaC_rest0(idx)=[];
[MetaC_rest0 idx]=rmoutliers(MetaC_rest0,'percentiles',[TL TH]);
MetaC_social0=MetaC_social0;
MetaC_social0(idx)=[];
boxplot([MetaC_rest0' MetaC_social0']);

a=MetaC_rest0;
b=MetaC_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'signrank');
pp=min(stats.pvals)


%%

a=EntropyQ_rest0-EntropyQ_social0;
b=EntropyC_rest0-EntropyC_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'ttest');
pp=min(stats.pvals)

a=MetaQ_rest0-MetaQ_social0;
b=MetaC_rest0-MetaC_social0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'ttest');
pp=min(stats.pvals)
