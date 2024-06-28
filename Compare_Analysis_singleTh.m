clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));
path5=[ '../LaplaceManifold/Sleep/'];
addpath(genpath(path5));

NSUB=1003; %1003
N=62;
Ttrain=800;

LATDIM=7;
Tau=0;

Isubdiag = find(tril(ones(N),-1));
IsubdiagL = find(tril(ones(LATDIM),-1));
IsubdiagT = find(tril(ones(1101),-1));

indexregion=[1:31 50:80];

%%% Define Training and test data
% Parameters of the data
TR=0.72;  % Repetition Time (seconds)
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter


load('hcp1003_REST1_LR_dbs80.mat');
epsilon=400;
Thorizont=1;

for sub=1:NSUB
    sub
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,:);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,50:end-50);

    zPhi=zscore(ts');
    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesA(:,t)=fcd(Isubdiag)';
    end
    FCDA=(EdgesA'*EdgesA)./(vecnorm(EdgesA)'*vecnorm(EdgesA));
    MetaA(sub)=0.5*(log(2*pi*var(FCDA(IsubdiagT))))+0.5;

    ts=zscore(ts,[],2);
    Tm=size(ts,2);
    Kmatrix=zeros(Tm,Tm);

    for i=1:Tm
        for j=1:Tm
            dij2=sum((ts(:,i)-ts(:,j)).^2);
            Kmatrix(i,j)=exp(-dij2/epsilon);
        end
    end

    Dmatrix=diag(sum(Kmatrix,2));
    Pmatrix=inv(Dmatrix)*Kmatrix;

    KmatrixTr=Kmatrix(1:Ttrain,1:Ttrain);
    DmatrixTr=diag(sum(KmatrixTr,2));
    PmatrixTr=inv(DmatrixTr)*KmatrixTr;
    [VV,LL]=eig(PmatrixTr);
    Phi=VV(:,2:LATDIM+1);

    %% CV
    TL=Tm-Ttrain;
    LAMBDA=LL(2:LATDIM+1,2:LATDIM+1).^Thorizont;
    Pcv=Kmatrix(Ttrain+1:end,1:Ttrain);
    for r=1:N
        tscvestimated=Pcv*Phi*inv(LAMBDA)*Phi'*ts(r,1:Ttrain)';
        tscve(r,:)=tscvestimated';
    end
    FCtrue=corrcoef(ts(:,Ttrain+1:end)');
    FCest=corrcoef(tscve');
    Corrfitt(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
    ERRfitt(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);
    FCtrue2(sub,:,:)=FCtrue;
    FCest2(sub,:,:)=FCest;
    %%
    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1).^Thorizont);
    zPhi=zscore(Phi);

    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesL(:,t)=fcd(IsubdiagL)';
    end

    FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
    Meta(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
end

FCtrueG=squeeze(mean(FCtrue2));
FCestG=squeeze(mean(FCest2));

%%%%%% Quantum
epsilon=300;
Thorizont=2;

for sub=1:NSUB
    sub
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,:);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,50:end-50);

    ts=zscore(ts,[],2);
    Tm=size(ts,2);
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

    Ptr_tTr=Ptr_t(1:Ttrain,1:Ttrain);
    DmatrixTr=diag(sum(Ptr_tTr,2));
    PmatrixTr=inv(DmatrixTr)*Ptr_tTr;

    [VV,LL]=eig(PmatrixTr);
    Phi=VV(:,2:LATDIM+1);

    %% CV
    TL=Tm-Ttrain;
    LAMBDA=LL(2:LATDIM+1,2:LATDIM+1);
    Pcv=Ptr_t(Ttrain+1:end,1:Ttrain);
    for r=1:N
        tscvestimated=Pcv*Phi*inv(LAMBDA)*Phi'*ts(r,1:Ttrain)';
        tscve(r,:)=tscvestimated';
    end
    FCtrue=corrcoef(ts(:,Ttrain+1:end)');
    FCest=corrcoef(tscve');
    CorrfittQ(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
    ERRfittQ(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);
    FCtrue2(sub,:,:)=FCtrue;
    FCest2(sub,:,:)=FCest;
    %%
    [VV,LL]=eig(Pmatrix);
    Phi=VV(:,2:LATDIM+1);
    Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

    zPhi=zscore(Phi);

    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesL(:,t)=fcd(IsubdiagL)';
    end

    FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
    MetaQ(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
end

FCtrueQ=squeeze(mean(FCtrue2));
FCestQ=squeeze(mean(FCest2));

%%   PCA

for sub=1:NSUB
    sub
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,:);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,50:end-50);

    ts=zscore(ts,[],2);

    [CoePCA,PhiPCA,llpca,tss,expl,mu]=pca(ts(:,1:Ttrain)');

    %% CV
    PhiPCAcv=ts(:,1+Ttrain:end)'*CoePCA;
    tscve=PhiPCAcv(:,1:LATDIM)*CoePCA(:,1:LATDIM)'+mu;

    FCtrue=corrcoef(ts(:,Ttrain+1:end)');
    FCest=corrcoef(tscve);
    CorrfittPCA(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
    ERRfittPCA(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);
    FCtrue2(sub,:,:)=FCtrue;
    FCest2(sub,:,:)=FCest;
    %%
    [CoePCA,Phi,llpca,tss,expl,mu]=pca(ts');
    zPhi=zscore(Phi);

    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesL(:,t)=fcd(IsubdiagL)';
    end

    FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
    MetaPCA(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
end

FCtruePCA=squeeze(mean(FCtrue2));
FCestPCA=squeeze(mean(FCest2));

for trial=1:100
    indsub=randperm(NSUB);
    CorrMeta(trial)=corr2(Meta(indsub(1:500)),MetaA(indsub(1:500)));
    CorrMetaQ(trial)=corr2(MetaQ(indsub(1:500)),MetaA(indsub(1:500)));
    CorrMetaPCA(trial)=corr2(MetaPCA(indsub(1:500)),MetaA(indsub(1:500)));
end


figure(1)
subplot(1,3,1);
scatter(MetaA,MetaPCA,'kx');
axis('square')
subplot(1,3,2);
scatter(MetaA,Meta,'bx');
axis('square')
subplot(1,3,3);
scatter(MetaA,MetaQ,'rx');
axis('square')

[ccAC pp]=corrcoef(MetaA,Meta)
[ccAQ pp]=corrcoef(MetaA,MetaQ)
[ccAPCA pp]=corrcoef(MetaA,MetaPCA)

ccAPCA(1,2)
ccAC(1,2)
ccAQ(1,2)


figure(2)
violinplot([CorrMetaPCA' CorrMeta' CorrMetaQ'])
axis('square')

ranksum(CorrMeta,CorrMetaQ)
ranksum(CorrMeta,CorrMetaPCA)
ranksum(CorrMetaQ,CorrMetaPCA)

% figure(3)
% violinplot([MetaPCA' Meta' MetaQ'])
% ranksum(Meta,MetaQ)
% ranksum(Meta,MetaPCA)
% ranksum(MetaQ,MetaPCA)

figure(4)
violinplot([CorrfittPCA' Corrfitt' CorrfittQ'])
axis('square')

ranksum(Corrfitt,CorrfittQ)
ranksum(Corrfitt,CorrfittPCA)
ranksum(CorrfittQ,CorrfittPCA)

figure(5)
violinplot([ERRfittPCA' ERRfitt' ERRfittQ'])
axis('square')

save results_analysis_single_LD7_Th2.mat CorrMeta CorrMetaQ CorrMetaPCA ...
    Corrfitt CorrfittQ CorrfittPCA ...
    ERRfitt ERRfittQ ERRfittPCA ...
    MetaA Meta MetaQ MetaPCA ...
    FCtrueG FCtrueQ FCtruePCA FCestG FCestQ FCestPCA;


