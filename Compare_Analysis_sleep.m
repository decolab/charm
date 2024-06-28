clear all;
load DataSleepW_N3.mat;

%% Example for comparison of two conditions....

N=90;
NSUB=15;

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


THS=[1 2 3 4 5];
epsilon=400;

for th=1:length(THS)
    Thorizont=THS(th)
    for sub=1:NSUB
        ts=TS_N3{sub};
        clear signal_filt tse;
        for seed=1:N
            ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
            signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
        end
        ts=signal_filt(:,10:end-10);

        zPhi=zscore(ts');
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesA(:,t)=fcd(Isubdiag)';
        end
        FCDA=dist(EdgesA);

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
        [VV,LL]=eig(Pmatrix);
        Phi=VV(:,2:LATDIM+1);

        %% reconstruction
        LAMBDA=LL(2:LATDIM+1,2:LATDIM+1).^Thorizont;
        for r=1:N
            tsestimated=Pmatrix*Phi*inv(LAMBDA)*Phi'*ts(r,:)';
            tse(r,:)=tsestimated';
        end
%         FCtrue=corrcoef(ts');
%         FCest=corrcoef(tse');
        ts2=ts';
        tse2=tse';
        for i=1:N
            for j=1:N
                [clag lags] = xcorr(ts2(:,i),ts2(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                FCtrue(i,j)=abs(clag(indx));
                [clag lags] = xcorr(tse2(:,i),tse2(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                FCest(i,j)=abs(clag(indx));
            end
        end
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr2(sub)=mean((FCtruevec-FCestvec).^2);
        %%

        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1).^Thorizont);

        zPhi=zscore(Phi);

        Covar=corrcoef(Phi);
        for i=1:LATDIM
            for j=1:LATDIM
                [clag lags] = xcorr(Phi(:,i),Phi(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                CovarShift(i,j)=abs(clag(indx));
            end
        end
        EntCov2(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));
%                 EntCov2(sub)=HShannon_kNN_k_estimation(zPhi',co);
        MeanCS2(sub)=mean(CovarShift(:));
        StdCS2(sub)=std(CovarShift(:));

        for t=1:Tm
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        Cofluctuations=sqrt(sum(Edges.^2));
        MeanCoFlu2(sub)=mean(Cofluctuations);
        StdCoFlu2(sub)=std(Cofluctuations);
        EntropyFlu2(sub)=0.5*(log(2*pi*var(Cofluctuations)))+0.5;
        %         EntropyFlu2(sub)=HShannon_kNN_k_estimation(Cofluctuations,co);

        FCD=dist(Edges);
        Meta2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
        %         fcdvec=FCD(find(tril(ones(size(FCD,1)),-1)));
        %         idxfcd=randperm(length(fcdvec));
        %         Meta2(sub)=HShannon_kNN_k_estimation(fcdvec(idxfcd(1:10000))',co);
        Fano2(sub)=var(FCD(:))/mean(FCD(:));

        [haux, paux, corrFCD2(sub)]=kstest2(FCDA(:),FCD(:));
    end

    EntCov(th)=mean(EntCov2);
    MeanCS(th)=mean(MeanCS2);
    StdCS(th)=mean(StdCS2);
    MeanCoFlu(th)=mean(MeanCoFlu2);
    StdCoFlu(th)=mean(StdCoFlu2);
    Meta(th)=mean(Meta2);
    Fano(th)=mean(Fano2);
    EntropyFlu(th)=mean(EntropyFlu2);
    corrFCD(th)=mean(corrFCD2);
    ErrFClr(th)=mean(ErrFClr2);

    EntCovs(th)=std(EntCov2);
    MeanCSs(th)=std(MeanCS2);
    StdCSs(th)=std(StdCS2);
    MeanCoFlus(th)=std(MeanCoFlu2);
    StdCoFlus(th)=std(StdCoFlu2);
    Metas(th)=std(Meta2);
    Fanos(th)=std(Fano2);
    EntropyFlus(th)=std(EntropyFlu2);
    corrFCDs(th)=std(corrFCD2);
    ErrFClrs(th)=std(ErrFClr2);

end

%%%%%% Quatum
THS=[1 2 3 4 5];
epsilon=300;

for th=1:length(THS)
    Thorizont=THS(th)
    for sub=1:NSUB
        ts=TS_N3{sub};
        clear signal_filt tse;
        for seed=1:N
            ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
            signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
        end
        ts=signal_filt(:,10:end-10);

        zPhi=zscore(ts');
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesA(:,t)=fcd(Isubdiag)';
        end
        FCDA=dist(EdgesA);

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

        [VV,LL]=eig(Pmatrix);
        Phi=VV(:,2:LATDIM+1);

        %% reconstruction
        LAMBDA=LL(2:LATDIM+1,2:LATDIM+1);
        for r=1:N
            tsestimated=Pmatrix*Phi*inv(LAMBDA)*Phi'*ts(r,:)';
            tse(r,:)=tsestimated';
        end
%         FCtrue=corrcoef(ts');
%         FCest=corrcoef(tse');
        ts2=ts';
        tse2=tse';
        for i=1:N
            for j=1:N
                [clag lags] = xcorr(ts2(:,i),ts2(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                FCtrue(i,j)=abs(clag(indx));
                [clag lags] = xcorr(tse2(:,i),tse2(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                FCest(i,j)=abs(clag(indx));
            end
        end
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr2(sub)=mean((FCtruevec-FCestvec).^2);
        %

        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));

        zPhi=zscore(Phi);

        Covar=corrcoef(Phi);
        for i=1:LATDIM
            for j=1:LATDIM
                [clag lags] = xcorr(Phi(:,i),Phi(:,j),Tau,'normalized');
                indx=find(lags==Tau);
                CovarShift(i,j)=abs(clag(indx));
            end
        end
        EntCov2(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));
%                 EntCov2(sub)=HShannon_kNN_k_estimation(zPhi',co);
        MeanCS2(sub)=mean(CovarShift(:));
        StdCS2(sub)=std(CovarShift(:));

        for t=1:Tm
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        Cofluctuations=sqrt(sum(Edges.^2));
        MeanCoFlu2(sub)=mean(Cofluctuations);
        StdCoFlu2(sub)=std(Cofluctuations);
        EntropyFlu2(sub)=0.5*(log(2*pi*var(Cofluctuations)))+0.5;
        %         EntropyFlu2(sub)=HShannon_kNN_k_estimation(Cofluctuations,co);

        FCD=dist(Edges);
        Meta2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
        %         fcdvec=FCD(find(tril(ones(size(FCD,1)),-1)));
        %         idxfcd=randperm(length(fcdvec));
        %         Meta2(sub)=HShannon_kNN_k_estimation(fcdvec(idxfcd(1:10000))',co);
        Fano2(sub)=var(FCD(:))/mean(FCD(:));
        [haux, paux, corrFCD2(sub)]=kstest2(FCDA(:),FCD(:));
    end

    EntCovQ(th)=mean(EntCov2);
    MeanCSQ(th)=mean(MeanCS2);
    StdCSQ(th)=mean(StdCS2);
    MeanCoFluQ(th)=mean(MeanCoFlu2);
    StdCoFluQ(th)=mean(StdCoFlu2);
    MetaQ(th)=mean(Meta2);
    FanoQ(th)=mean(Fano2);
    EntropyFluQ(th)=mean(EntropyFlu2);
    corrFCDq(th)=mean(corrFCD2);
    ErrFClrq(th)=mean(ErrFClr2);


    EntCovQs(th)=std(EntCov2);
    MeanCSQs(th)=std(MeanCS2);
    StdCSQs(th)=std(StdCS2);
    MeanCoFluQs(th)=std(MeanCoFlu2);
    StdCoFluQs(th)=std(StdCoFlu2);
    MetaQs(th)=std(Meta2);
    FanoQs(th)=std(Fano2);
    EntropyFluQs(th)=std(EntropyFlu2);
    corrFCDqs(th)=std(corrFCD2);
    ErrFClrqs(th)=std(ErrFClr2);

end

%%   PCA

for sub=1:NSUB
    ts=TS_N3{sub};
    clear signal_filt tse;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
    end
    ts=signal_filt(:,10:end-10);

    zPhi=zscore(ts');
    for t=1:size(zPhi,1)
        fcd=zPhi(t,:)'*zPhi(t,:);
        EdgesA(:,t)=fcd(Isubdiag)';
    end
    FCDA=dist(EdgesA);

    ts=zscore(ts,[],2);
    Tm=size(ts,2);

    [CoePCA,PhiPCA,llpca,tss,expl,mu]=pca(ts');

    %% reconstruction
    PhiPCAcv=ts'*CoePCA;
    tse=PhiPCAcv(:,1:LATDIM)*CoePCA(:,1:LATDIM)'+mu;

    %         FCtrue=corrcoef(ts');
    %         FCest=corrcoef(tse);
    ts2=ts';
    tse2=tse;
    for i=1:N
        for j=1:N
            [clag lags] = xcorr(ts2(:,i),ts2(:,j),Tau,'normalized');
            indx=find(lags==Tau);
            FCtrue(i,j)=abs(clag(indx));
            [clag lags] = xcorr(tse2(:,i),tse2(:,j),Tau,'normalized');
            indx=find(lags==Tau);
            FCest(i,j)=abs(clag(indx));
        end
    end
    FCtruevec=FCtrue(:);
    FCestvec=FCest(:);
    FCtruevec(find(isnan(FCtruevec)))=[];
    FCestvec(find(isnan(FCestvec)))=[];
    ErrFClr2(sub)=mean((FCtruevec-FCestvec).^2);
    %

    Phi=PhiPCA(:,1:LATDIM)*diag(llpca(1:LATDIM));

    zPhi=zscore(Phi);

    Covar=corrcoef(Phi);
    for i=1:LATDIM
        for j=1:LATDIM
            [clag lags] = xcorr(Phi(:,i),Phi(:,j),Tau,'normalized');
            indx=find(lags==Tau);
            CovarShift(i,j)=abs(clag(indx));
        end
    end
    EntCov2(sub)=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));
%         EntCov2(sub)=HShannon_kNN_k_estimation(zPhi',co);
    MeanCS2(sub)=mean(CovarShift(:));
    StdCS2(sub)=std(CovarShift(:));

    for t=1:Tm
        fcd=zPhi(t,:)'*zPhi(t,:);
        Edges(:,t)=fcd(Isubdiag)';
    end
    Cofluctuations=sqrt(sum(Edges.^2));
    MeanCoFlu2(sub)=mean(Cofluctuations);
    StdCoFlu2(sub)=std(Cofluctuations);
    EntropyFlu2(sub)=0.5*(log(2*pi*var(Cofluctuations)))+0.5;
%         EntropyFlu2(sub)=HShannon_kNN_k_estimation(Cofluctuations,co);

    FCD=dist(Edges);
    Meta2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
    %     fcdvec=FCD(find(tril(ones(size(FCD,1)),-1)));
    %     idxfcd=randperm(length(fcdvec));
    %     Meta2(sub)=HShannon_kNN_k_estimation(fcdvec(idxfcd(1:10000))',co);
    Fano2(sub)=var(FCD(:))/mean(FCD(:));
    [haux, paux, corrFCDPCA2(sub)]=kstest2(FCDA(:),FCD(:));
end

EntCovPCA=mean(EntCov2);
MeanCSPCA=mean(MeanCS2);
StdCSPCA=mean(StdCS2);
MeanCoFluPCA=mean(MeanCoFlu2);
StdCoFluPCA=mean(StdCoFlu2);
MetaPCA=mean(Meta2);
FanoPCA=mean(Fano2);
EntropyFluPCA=mean(EntropyFlu2);
corrFCDPCA=mean(corrFCDPCA2);
ErrFClrPCA=mean(ErrFClr2);


EntCovPCAs=std(EntCov2);
MeanCSPCAs=std(MeanCS2);
StdCSPCAs=std(StdCS2);
MeanCoFluPCAs=std(MeanCoFlu2);
StdCoFluPCAs=std(StdCoFlu2);
MetaPCAs=std(Meta2);
FanoPCAs=std(Fano2);
EntropyFluPCAs=std(EntropyFlu2);
corrFCDPCAs=std(corrFCDPCA2);
ErrFClrPCAs=std(ErrFClr2);

figure(1)
shadedErrorBar(THS,EntCov,EntCovs,'k',0.7);
hold on;
shadedErrorBar(THS,EntCovQ,EntCovQs,'r',0.7);
shadedErrorBar(THS,EntCovPCA*ones(1,length(THS)),EntCovPCAs*ones(1,length(THS)),'b-',0.5);

figure(2)
shadedErrorBar(THS,ErrFClr,ErrFClrs,'k',0.7);
hold on;
shadedErrorBar(THS,ErrFClrq,ErrFClrqs,'r',0.7);
shadedErrorBar(THS,ErrFClrPCA*ones(1,length(THS)),ErrFClrPCAs*ones(1,length(THS)),'b-',0.5);

figure(3)
shadedErrorBar(THS,corrFCD,corrFCDs,'k',0.7);
hold on;
shadedErrorBar(THS,corrFCDq,corrFCDqs,'r',0.7);
shadedErrorBar(THS,corrFCDPCA*ones(1,length(THS)),corrFCDPCAs*ones(1,length(THS)),'b-',0.5);

save results_analysis_sleepN3.mat ...
    EntropyFlu EntropyFlus EntropyFluQ EntropyFluQs EntropyFluPCA EntropyFluPCAs ...
    EntCov EntCovs EntCovQ EntCovQs EntCovPCA EntCovPCAs ...
    MeanCS MeanCSs MeanCSQ MeanCSQs MeanCSPCA MeanCSPCAs ...
    Fano Fanos FanoQ FanoQs FanoPCA FanoPCAs ...
    corrFCD corrFCDq corrFCDPCA corrFCDs corrFCDqs corrFCDPCAs ...
    ErrFClr ErrFClrq ErrFClrPCA ErrFClrs ErrFClrqs ErrFClrPCAs ...
    Meta Metas MetaQ MetaQs MetaPCA MetaPCAs;


% 
% figure(2)
% shadedErrorBar(THS,MeanCS,MeanCSs,'k',0.7);
% hold on;
% shadedErrorBar(THS,MeanCSQ,MeanCSQs,'r',0.7);
% shadedErrorBar(THS,MeanCSPCA*ones(1,length(THS)),MeanCSPCAs*ones(1,length(THS)),'b-',0.5);
% 
% figure(3)
% shadedErrorBar(THS,EntropyFlu,EntropyFlus,'k',0.7);
% hold on;
% shadedErrorBar(THS,EntropyFluQ,EntropyFluQs,'r',0.7);
% shadedErrorBar(THS,EntropyFluPCA*ones(1,length(THS)),EntropyFluPCAs*ones(1,length(THS)),'b-',0.5);

% figure(1)
% plot(EntCov,'k')
% hold on;
% plot(EntCovQ,'r')
% 
% figure(2)
% plot(MeanCS,'k')
% hold on;
% plot(MeanCSQ,'r')
% 
% figure(3)
% plot(StdCS,'k')
% hold on;
% plot(StdCSQ,'r')
% 
% figure(4)
% plot(MeanCoFlu,'k')
% hold on;
% plot(MeanCoFluQ,'r')
% 
% figure(5)
% plot(StdCoFlu,'k')
% hold on;
% plot(StdCoFluQ,'r')
% 
% figure(6)
% plot(EntropyFlu,'k')
% hold on;
% plot(EntropyFluQ,'r')
% 
% figure(7)
% plot(Meta,'k')
% hold on;
% plot(MetaQ,'r')
% 
% figure(8)
% plot(Fano,'k')
% hold on;
% plot(FanoQ,'r')
