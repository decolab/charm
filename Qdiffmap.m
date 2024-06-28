clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));


NSUB=1003; %1003
N=62;

LATDIM=7;
Ttrain=800; %550;
Tau=3;

Isubdiag = find(tril(ones(N),-1));

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
EPS=[1 10 20 30 50 100 150 200 300 400];
THS=[1 2 3 4 5 6];

for ep=1:length(EPS)
    for th=1:length(THS)
        epsilon=EPS(ep)
        Thorizont=THS(th)
        for sub=1:NSUB
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
                    Kmatrix(i,j)=exp(-dij2/epsilon);
                end
            end

%             Dmatrix=diag(sum(Kmatrix,2));
%             Ktr=inv(Dmatrix)*Kmatrix*inv(Dmatrix);

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

            FCtautrue=corr(ts(:,Ttrain+1:Tm-Tau)',ts(:,Ttrain+1+Tau:Tm)');
            FCtauest=corr(tscve(:,1:TL-Tau)',tscve(:,1+Tau:TL)');
            CORRtaufitt2(sub)=corr2(FCtautrue(:),FCtauest(:));
            ERRtaufitt2(sub)=mean((FCtautrue(:)-FCtauest(:)).^2);

            [haux,paux,ErrfittFC2(sub)]=kstest2(FCtrue(Isubdiag),FCest(Isubdiag));

            zPhi=zscore(ts(:,Ttrain+1:end)');
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edgest(:,t)=fcd(Isubdiag)';
            end
            FCDt=dist(Edgest);
            zPhi=zscore(tscve');
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edgese(:,t)=fcd(Isubdiag)';
            end
            FCDe=dist(Edgese);
            [haux,paux,ErrfittFCD2(sub)]=kstest2(FCDt(:),FCDe(:));
        end

        CorrFitt(ep,th)=mean(CORRfitt2)
        ErrFitt(ep,th)=mean(ERRfitt2);
        CorrFittTau(ep,th)=mean(CORRtaufitt2)
        ErrFittTau(ep,th)=mean(ERRtaufitt2);
        ErrFittFC(ep,th)=mean(ErrfittFC2);
        ErrFittFCD(ep,th)=mean(ErrfittFCD2);
    end
end


%%%%%% Quatum
% EPS=[0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100];
EPS=[1 10 20 30 50 100 150 200 300];
THS=[1 2 3 4 5 6];

for ep=1:length(EPS)
    for th=1:length(THS)
        epsilon=EPS(ep)
        Thorizont=THS(th)
        for sub=1:NSUB
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
            tscve=real(tscve);
            FCtrue=corrcoef(ts(:,Ttrain+1:end)');
            FCest=corrcoef(tscve');
            CORRfitt2q(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
            ERRfitt2q(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);
            FCtautrue=corr(ts(:,Ttrain+1:Tm-Tau)',ts(:,Ttrain+1+Tau:Tm)');
            FCtauest=corr(tscve(:,1:TL-Tau)',tscve(:,1+Tau:TL)');
            CORRtaufitt2q(sub)=corr2(FCtautrue(:),FCtauest(:));
            ERRtaufitt2q(sub)=mean((FCtautrue(:)-FCtauest(:)).^2);

            [haux,paux,ErrfittFCq2(sub)]=kstest2(FCtrue(Isubdiag),FCest(Isubdiag));

            zPhi=zscore(ts(:,Ttrain+1:end)');
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edgest(:,t)=fcd(Isubdiag)';
            end
            FCDt=dist(Edgest);
            zPhi=zscore(tscve');
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edgese(:,t)=fcd(Isubdiag)';
            end
            FCDe=dist(Edgese);
            [haux,paux,ErrfittFCDq2(sub)]=kstest2(FCDt(:),FCDe(:));
        end

        CorrFittq(ep,th)=mean(CORRfitt2q)
        ErrFittq(ep,th)=mean(ERRfitt2q);
        CorrFittTauq(ep,th)=mean(CORRtaufitt2q)
        ErrFittTauq(ep,th)=mean(ERRtaufitt2q);
        ErrFittFCq(ep,th)=mean(ErrfittFCq2);
        ErrFittFCDq(ep,th)=mean(ErrfittFCDq2);
    end
end


%%   PCA

for sub=1:NSUB
    ts=subject{sub}.dbs80ts;
    ts=ts(indexregion,:);
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,50:end-50);
    ts=zscore(ts,[],2);

    [CoePCA,PhiPCA,llpca,tss,exp,mu]=pca(ts(:,1:Ttrain)');

    %% CV
    PhiPCAcv=ts(:,1+Ttrain:end)'*CoePCA;
    tscve=PhiPCAcv(:,1:LATDIM)*CoePCA(:,1:LATDIM)'+mu;

    FCtrue=corrcoef(ts(:,Ttrain+1:end)');
    FCest=corrcoef(tscve);
    CORRfitt2pca(sub)=corr2(FCtrue(Isubdiag),FCest(Isubdiag));
    ERRfitt2pca(sub)=mean((FCtrue(Isubdiag)-FCest(Isubdiag)).^2);
    FCtautrue=corr(ts(:,Ttrain+1:Tm-Tau)',ts(:,Ttrain+1+Tau:Tm)');
    FCtauest=corr(tscve(1:TL-Tau,:),tscve(1+Tau:TL,:));
    CORRtaufitt2pca(sub)=corr2(FCtautrue(:),FCtauest(:));
    ERRtaufitt2pca(sub)=mean((FCtautrue(:)-FCtauest(:)).^2);

    [haux,paux,ErrfittFCPCA2(sub)]=kstest2(FCtrue(Isubdiag),FCest(Isubdiag));

    zPhi=zscore(ts(:,Ttrain+1:end)');
    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edgest(:,t)=fcd(Isubdiag)';
    end
    FCDt=dist(Edgest);
    zPhi=zscore(tscve');
    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edgese(:,t)=fcd(Isubdiag)';
    end
    FCDe=dist(Edgese);
    [haux,paux,ErrfittFCDPCA2(sub)]=kstest2(FCDt(:),FCDe(:));
end

CorrFittPCA=mean(CORRfitt2pca)
ErrFittPCA=mean(ERRfitt2pca);
CorrFittTauPCA=mean(CORRtaufitt2pca)
ErrFittTauPCA=mean(ERRtaufitt2pca);
ErrFittFCPCA=mean(ErrfittFCPCA2);
ErrFittFCDPCA=mean(ErrfittFCDPCA2);

figure(1)
subplot(1,2,1)
imagesc(CorrFitt(2:10,:));
axis('square')
subplot(1,2,2)
imagesc(CorrFittq(2:9,:));
axis('square')

CorrFittPCA

figure(2)
subplot(1,2,1)
imagesc(ErrFitt(2:10,:));
zlim([0 0.25]);
axis('square')
subplot(1,2,2)
imagesc(ErrFittq(2:9,:));
zlim([0 0.25]);
axis('square')

ErrFittPCA

save results_diffmaps.mat CorrFittPCA CorrFittTauPCA ErrFittPCA ErrFittTauPCA ...
    CorrFitt CorrFittTau ErrFitt ErrFittTau ...
    CorrFittq CorrFittTauq ErrFittq ErrFittTauq ...
    ErrFittFC ErrFittFCD ErrFittFCq ErrFittFCDq ErrFittFCPCA ErrFittFCDPCA;
