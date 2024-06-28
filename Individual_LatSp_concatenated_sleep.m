clear all;
load DataSleepW_N3.mat;

%% Example for comparison of two conditions....

N=80;
NSUB=15;

indexregions=[1:40 51:90];

LATDIM=7;
Tau=0;

Isubdiag = find(tril(ones(LATDIM),-1));

TR=2.08;  % Repetition Time (seconds)
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

load DataSleepW_N3.mat;

for sub=1:NSUB
    twake=size(TS_W{sub},2);
    tn3=size(TS_N3{sub},2);
    timets(sub)=min(twake,tn3);
end

for sub=1:NSUB
    sub
    clear signal_filt ts Edges;
    ts=TS_W{sub};
    ts=ts(indexregions,1:timets(sub));
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    tsr=signal_filt(:,10:end-10);

    ts=TS_N3{sub};
    ts=ts(indexregions,1:timets(sub));
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    tss=signal_filt(:,10:end-10);
    ts=[tsr tss];

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

    Covar=corrcoef(Phi(1:Tm/2,:));
    EntropyC_rest(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));
    Covar=corrcoef(Phi(Tm/2+1:end,:));
    EntropyC_task(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi))); 

    IsubdiagT = find(tril(ones(Tm/2),-1));

    zPhi=zscore(Phi(1:Tm/2,:));

    for t=1:Tm/2
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=(Edges'*Edges)./(vecnorm(Edges)'*vecnorm(Edges));
    MetaC_rest(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

    zPhi=zscore(Phi(Tm/2+1:Tm,:));

    for t=1:Tm/2
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=(Edges'*Edges)./(vecnorm(Edges)'*vecnorm(Edges));
    MetaC_task(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

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
    Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));

    Covar=corrcoef(Phi(1:Tm/2,:));
    EntropyQ_rest(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));
    Covar=corrcoef(Phi(Tm/2+1:end,:));
    EntropyQ_task(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi))); 

    zPhi=zscore(Phi(1:Tm/2,:));

    for t=1:Tm/2
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=(Edges'*Edges)./(vecnorm(Edges)'*vecnorm(Edges));
    MetaQ_rest(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

    zPhi=zscore(Phi(Tm/2+1:Tm,:));
    
    for t=1:Tm/2
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    FCD=(Edges'*Edges)./(vecnorm(Edges)'*vecnorm(Edges));
    MetaQ_task(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

end

%%
TL=0;
TH=100;

figure(1)
subplot(1,2,1)
[EntropyQ_task0 idx]=rmoutliers(EntropyQ_task,'percentiles',[TL TH]);
EntropyQ_task0=EntropyQ_task;
EntropyQ_task0(idx)=[];
[EntropyQ_rest0 idx]=rmoutliers(EntropyQ_rest,'percentiles',[TL TH]);
EntropyQ_rest0=EntropyQ_rest;
EntropyQ_rest0(idx)=[];
boxplot([EntropyQ_rest0' EntropyQ_task0']);

a=EntropyQ_rest0;
b=EntropyQ_task0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],1000,0.01,'ttest2');
pp=min(stats.pvals)

subplot(1,2,2)
[EntropyC_task0 idx]=rmoutliers(EntropyC_task,'percentiles',[TL TH]);
EntropyC_task0=EntropyC_task;
EntropyC_task0(idx)=[];
[EntropyC_rest0 idx]=rmoutliers(EntropyC_rest,'percentiles',[TL TH]);
EntropyC_rest0=EntropyC_rest;
EntropyC_rest0(idx)=[];
boxplot([EntropyC_rest0' EntropyC_task0']);

a=EntropyC_rest0;
b=EntropyC_task0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'ttest2');
pp=min(stats.pvals)


figure(2)
TL=0;
TH=100;
subplot(1,2,1)
[MetaQ_task0 idx]=rmoutliers(MetaQ_task,'percentiles',[TL TH]);
MetaQ_task0=MetaQ_task;
MetaQ_task0(idx)=[];
[MetaQ_rest0 idx]=rmoutliers(MetaQ_rest,'percentiles',[TL TH]);
MetaQ_rest0=MetaQ_rest;
MetaQ_rest0(idx)=[];
boxplot([MetaQ_rest0' MetaQ_task0']);

a=MetaQ_rest0;
b=MetaQ_task0;
signrank(a,b)


subplot(1,2,2)
[MetaC_task0 idx]=rmoutliers(MetaC_task,'percentiles',[TL TH]);
MetaC_task0=MetaC_task;
MetaC_task0(idx)=[];
[MetaC_rest0 idx]=rmoutliers(MetaC_rest,'percentiles',[TL TH]);
MetaC_rest0=MetaC_rest;
MetaC_rest0(idx)=[];
boxplot([MetaC_rest0' MetaC_task0']);

a=MetaC_rest0;
b=MetaC_task0;
signrank(a,b)

%

a=EntropyQ_rest0-EntropyQ_task0;
b=EntropyC_rest0-EntropyC_task0;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'ttest');
pp=min(stats.pvals)

a=(MetaQ_rest0-MetaQ_task0);
b=(MetaC_rest0-MetaC_task0);
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],10000,0.01,'ttest');
pp=min(stats.pvals)

figure(3)
boxplot([a' b']);

%%
% 
% figure(3)
% scatter(Phi(1:Tm/2,1),Phi(1:Tm/2,2),'k');
% hold on;
% scatter(Phi(Tm/2+1:end,1),Phi(Tm/2+1:end,2),'r');
% 
% figure(4)
% Y=tsne(Phi,'Algorithm','exact','Distance','mahalanobis');
% scatter(Y(1:Tm/2,1),Y(1:Tm/2,2),'k');
% hold on;
% scatter(Y(Tm/2+1:end,1),Y(Tm/2+1:end,2),'r');

save results_individual_concat_sleep.mat MetaQ_rest0 MetaC_rest0 EntropyQ_rest0 EntropyC_rest0 ...
    EntropyQ_task0 EntropyC_task0 MetaQ_task0 MetaC_task0;