clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));

NSUB=30; 
N=62;

Tau=0;

LATDIM=7;

Isubdiag = find(tril(ones(LATDIM),-1));

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

indexregion=[1:31 50:80];

load cog_dk80.mat;
dk80cog=dk80cog(indexregion,:);
for i=1:N
    for j=1:N
        SCdist(i,j)=sqrt(sum((dk80cog(i,:)-dk80cog(j,:)).^2));
        if SCdist(i,j)>0
            SCmasklr(i,j)=1;
        else
            SCmasklr(i,j)=NaN;
        end
    end
end

load results_hopf_TSsim.mat;
epsilon=300;

%%%%%% Quatum
ng=1;
nsub=1;
for G=G_range
    G
    for sub=1:NSUB
        ts=squeeze(TSsim(ng,sub,:,:));
        for seed=1:N
            ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
            signal_filt(seed,:)=(filtfilt(bfilt,afilt,ts(seed,:)));
        end
        ts1=signal_filt(:,50:end-50);
        ts=zscore(ts1,[],2);
        Tm=size(ts,2);
        Kmatrix=zeros(Tm,Tm);

        for i=1:Tm
            for j=1:Tm
                dij2=sum((ts(:,i)-ts(:,j)).^2);
                Kmatrix(i,j)=exp(complex(0,1)*dij2/epsilon);
            end
        end

        Ktr_t=Kmatrix;
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
        FCtrue=FCtrue.*SCmasklr;
        FCest=FCest.*SCmasklr;
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr1=mean((FCtruevec-FCestvec).^2);
        %%

        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));
        Covar=corrcoef(Phi);
        EntCovQ1=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

        %%
        zPhi=zscore(Phi);
        for t=1:Tm
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        FCD=dist(Edges);
        EdgeMeta1=0.5*(log(2*pi*var(FCD(:))))+0.5;
        %%

        Ktr_t=Kmatrix^2;
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
        FCtrue=FCtrue.*SCmasklr;
        FCest=FCest.*SCmasklr;
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr2=mean((FCtruevec-FCestvec).^2);
        %%

        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));
        Covar=corrcoef(Phi);
        EntCovQ2=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

        %%
        zPhi=zscore(Phi);
        for t=1:Tm
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        FCD=dist(Edges);
        EdgeMeta2=0.5*(log(2*pi*var(FCD(:))))+0.5;
        %%

        Ktr_t=Kmatrix^3;
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
        FCtrue=FCtrue.*SCmasklr;
        FCest=FCest.*SCmasklr;
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr3=mean((FCtruevec-FCestvec).^2);
        %%

        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));
        Covar=corrcoef(Phi);
        EntCovQ3=0.5*(log(det(Covar))+LATDIM*(1+log(2*pi)));

        %%
        zPhi=zscore(Phi);
        for t=1:Tm
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        FCD=dist(Edges);
        EdgeMeta3=0.5*(log(2*pi*var(FCD(:))))+0.5;
       
        %%

        Ktr_t=Kmatrix^5;
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
        FCtrue=FCtrue.*SCmasklr;
        FCest=FCest.*SCmasklr;
        FCtruevec=FCtrue(:);
        FCestvec=FCest(:);
        FCtruevec(find(isnan(FCtruevec)))=[];
        FCestvec(find(isnan(FCestvec)))=[];
        ErrFClr5=mean((FCtruevec-FCestvec).^2);
        %%

        DeltaEntCovQ13_2(sub)=EntCovQ1-EntCovQ3;
        Convexity2(sub)=((EntCovQ1-EntCovQ3)/2-(EntCovQ2-EntCovQ3))/((EntCovQ1-EntCovQ3)/2);

        FCsim=corrcoef(ts1');
        GlobalEff2(sub)=efficiency_wei(abs(FCsim-diag(diag(FCsim))));

        DeltaEntCovQ13sub(nsub)=DeltaEntCovQ13_2(sub);
        GlobalEffsub(nsub)=GlobalEff2(sub);
        Convexitysub(nsub)=Convexity2(sub);

        DeltaEdgeMeta13_2(sub)=EdgeMeta1-EdgeMeta3;
        EdgeConvexity2(sub)=((EdgeMeta1-EdgeMeta3)/2-(EdgeMeta2-EdgeMeta3))/((EdgeMeta1-EdgeMeta3)/2);

        DeltaFClr_2(sub)=ErrFClr1-ErrFClr3;
        FClrConvexity2(sub)=((ErrFClr1-ErrFClr3)/2-(ErrFClr2-ErrFClr3))/((ErrFClr1-ErrFClr3)/2);

        nsub=nsub+1;
    end

    DeltaEntCovQ13(ng)=mean(DeltaEntCovQ13_2)
    GlobalEff(ng)=mean(GlobalEff2)
    Convexity(ng)=mean(Convexity2)

    DeltaEdgeMeta13_2_z=demean(detrend(DeltaEdgeMeta13_2));
    [DeltaEdgeMeta13_2_z idx]=rmoutliers(DeltaEdgeMeta13_2_z,'percentiles',[10 90]);
    DeltaEdgeMeta13_2(idx)=[];
    DeltaEdgeMeta13(ng)=mean(DeltaEdgeMeta13_2)
    
    EdgeConvexity2_z=demean(detrend(EdgeConvexity2));
    [EdgeConvexity2_z idx]=rmoutliers(EdgeConvexity2_z,'percentiles',[10 90]);
    EdgeConvexity2(idx)=[];
    EdgeConvexity(ng)=mean(EdgeConvexity2)

    DeltaFClr_2_z=demean(detrend(DeltaFClr_2));
    [DeltaFClr_2_z idx]=rmoutliers(DeltaFClr_2_z,'percentiles',[10 90]);
    DeltaFClr_2(idx)=[];
    DeltaFClr15(ng)=mean(DeltaFClr_2)

    FClrConvexity2_z=demean(detrend(FClrConvexity2));
    [FClrConvexity2_z idx]=rmoutliers(FClrConvexity2_z,'percentiles',[10 90]);
    FClrConvexity2(idx)=[];
    FClrConvexity(ng)=mean(FClrConvexity2)

    ng=ng+1;
end

% figure(1)
% scatter(GlobalEff,DeltaEntCovQ13)
% 
% figure(2)
% scatter(GlobalEff,Convexity)
% 
% 
% corr2(GlobalEff,DeltaEntCovQ13)
% corr2(GlobalEff,Convexity)
% 
% mdl=fitlm(GlobalEff,DeltaEntCovQ13,'quadratic');
% mdl.Rsquared.Ordinary
% 
% mdl=fitlm(GlobalEff,Convexity,'quadratic');
% mdl.Rsquared.Ordinary

%%
% 
figure(1)
scatter(GlobalEff,DeltaEdgeMeta13)

figure(2)
scatter(GlobalEff,EdgeConvexity)


corr2(GlobalEff,DeltaEdgeMeta13)
corr2(GlobalEff,EdgeConvexity)

mdl=fitlm(GlobalEff,DeltaEdgeMeta13,'quadratic');
mdl.Rsquared.Ordinary

mdl=fitlm(GlobalEff0,EdgeConvexity0,'quadratic');
mdl.Rsquared.Ordinary

%%


figure(3)
scatter(GlobalEff,DeltaFClr15)

figure(4)
scatter(GlobalEff,FClrConvexity)


corr2(GlobalEff,DeltaFClr15)
corr2(GlobalEff,FClrConvexity)

mdl=fitlm(GlobalEff,DeltaFClr15,'quadratic');
mdl.Rsquared.Ordinary

mdl=fitlm(GlobalEff0,FClrConvexity0,'quadratic');
mdl.Rsquared.Ordinary


save results_efficiency_simul.mat GlobalEff DeltaEntCovQ13 Convexity ...
    DeltaEdgeMeta13 EdgeConvexity DeltaFClr15 FClrConvexity;