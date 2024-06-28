clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));
path5=[ '../LaplaceManifold/Sleep/'];
addpath(genpath(path5));


NSUB=1003;
N=62;
Ttrain=800;
LATDIM=7;

Isubdiag = find(tril(ones(N),-1));
IsubdiagL = find(tril(ones(LATDIM),-1));
IsubdiagT = find(tril(ones(1101),-1));

indexregion=[1:31 50:80];

TR=0.72;  % Repetition Time (seconds)
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter


load('hcp1003_REST1_LR_dbs80.mat');
THS=[1 2 3 4 5];
epsilon=400;

for th=1:length(THS)
    Thorizont=THS(th)
    for sub=1:NSUB
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
        CORRfitt2(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
        ERRfitt2(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);

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

    for trial=1:100
        indsub=randperm(NSUB);
        corrmeta2(trial)=corr2(Meta(indsub(1:500)),MetaA(indsub(1:500)));
    end
    corrMeta(th)=mean(corrmeta2);
    corrMetas(th)=std(corrmeta2);

    LatMeta(th)=mean(Meta);
    LatMetas(th)=std(Meta);

    Corrfitt(th)=mean(CORRfitt2);
    Errfitt(th)=mean(ERRfitt2);

    Corrfitts(th)=std(CORRfitt2);
    Errfitts(th)=std(ERRfitt2);

end

%%%%%% Quantum
THS=[1 2 3 4 5];
epsilon=300;

for th=1:length(THS)
    Thorizont=THS(th)
    for sub=1:NSUB
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
        CORRfitt2(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
        ERRfitt2(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);

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
        Meta(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
    end

    for trial=1:100
        indsub=randperm(NSUB);
        corrmeta2(trial)=corr2(Meta(indsub(1:500)),MetaA(indsub(1:500)));
    end
    corrMetaQ(th)=mean(corrmeta2);
    corrMetaQs(th)=std(corrmeta2);

    LatMetaQ(th)=mean(Meta);
    LatMetaQs(th)=std(Meta);

    CorrfittQ(th)=mean(CORRfitt2);
    ErrfittQ(th)=mean(ERRfitt2);

    CorrfittQs(th)=std(CORRfitt2);
    ErrfittQs(th)=std(ERRfitt2);
end

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
    Tm=size(ts,2);

    zPhi=zscore(ts');
    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesA(:,t)=fcd(Isubdiag)';
    end
    FCDA=(EdgesA'*EdgesA)./(vecnorm(EdgesA)'*vecnorm(EdgesA));
    MetaA(sub)=0.5*(log(2*pi*var(FCDA(IsubdiagT))))+0.5;

    ts=zscore(ts,[],2);

    [CoePCA,PhiPCA,llpca,tss,expl,mu]=pca(ts(:,1:Ttrain)');

    %% CV
    TL=Tm-Ttrain;

    PhiPCAcv=ts(:,1+Ttrain:end)'*CoePCA;
    tscve=PhiPCAcv(:,1:LATDIM)*CoePCA(:,1:LATDIM)'+mu;

    FCtrue=corrcoef(ts(:,Ttrain+1:end)');
    FCest=corrcoef(tscve);
    CORRfitt2(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
    ERRfitt2(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);

    %%
    [CoePCA,Phi,llpca,tss,expl,mu]=pca(ts');
    zPhi=zscore(Phi);

    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesL(:,t)=fcd(IsubdiagL)';
    end

    FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
    Meta(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
end

for trial=1:100
    indsub=randperm(NSUB);
    corrmeta2(trial)=corr2(Meta(indsub(1:500)),MetaA(indsub(1:500)));
end
for th=1:length(THS)
    corrMetaPCA(th)=mean(corrmeta2);
    corrMetaPCAs(th)=std(corrmeta2);
    LatMetaPCA(th)=mean(Meta);
    LatMetaPCAs(th)=std(Meta);
    CorrfittPCA(th)=mean(CORRfitt2);
    ErrfittPCA(th)=mean(ERRfitt2);
    CorrfittPCAs(th)=std(CORRfitt2);
    ErrfittPCAs(th)=std(ERRfitt2);
end

figure(1)
shadedErrorBar(THS,corrMeta,corrMetas,'k',0.7);
hold on;
shadedErrorBar(THS,corrMetaQ,corrMetaQs,'r',0.7);
shadedErrorBar(THS,corrMetaPCA,corrMetaPCAs,'b-',0.7);
axis('square');

figure(2)
shadedErrorBar(THS,Corrfitt,Corrfitts,'k',0.7);
hold on;
shadedErrorBar(THS,CorrfittQ,CorrfittQs,'r',0.7);
shadedErrorBar(THS,CorrfittPCA,CorrfittPCAs,'b-',0.7);
axis('square');

figure(3)
shadedErrorBar(THS,Errfitt,Errfitts,'k',0.7);
hold on;
shadedErrorBar(THS,ErrfittQ,ErrfittQs,'r',0.7);
shadedErrorBar(THS,ErrfittPCA,ErrfittPCAs,'b-',0.7);

%
% figure(7)
% shadedErrorBar(THS,LatMeta,LatMetas,'k',0.7);
% hold on;
% shadedErrorBar(THS,LatMetaQ,LatMetaQs,'r',0.7);
% shadedErrorBar(THS,LatMetaPCA,LatMetaPCAs,'b-',0.7);

save results_analysis_LD7.mat corrMeta corrMetaQ corrMetaPCA THS ...
    corrMetas corrMetaQs corrMetaPCAs ...
    Corrfitt Corrfitts CorrfittQ CorrfittQs CorrfittPCA CorrfittPCAs ...
    Errfitt Errfitts ErrfittQ ErrfittQs ErrfittPCA ErrfittPCAs ...
    LatMeta LatMetas LatMetaQ LatMetaQs LatMetaPCA LatMetaPCAs;