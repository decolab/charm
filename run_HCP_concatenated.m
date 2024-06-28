clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));
path5=[ '../LaplaceManifold/Sleep/'];
addpath(genpath(path5));

%% Example for comparison of two conditions....

Tmax=274;  %% emotion 175   social 274

NSUB=100;
NTRAIN=90;
N=62;

LATDIM=7;
Isubdiag = find(tril(ones(LATDIM),-1));

indexregion=[1:31 50:80];

FULLWIN=1;

LATDIM=7;
kfold=10000;
Npatterns=100;
WIN=30;

TR=0.72;  % Repetition Time (seconds)
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

load('hcpunrelated100_resting_dbs80.mat');
subject_rest=subject;
load('hcpunrelated100_socialtask_dbs80.mat');
subject_task=subject;

%% Classic Diff Map
Thorizont=1;
epsilon=400;

TS=[];
lastupidx=0;
for sub=1:NSUB  % over subjects
    ts=subject_rest{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx(sub)=1+lastupidx;
    nsubupidx(sub)=nsublowidx(sub)+size(ts,2)-1;
    lastupidx=nsubupidx(sub);
    TS=[TS ts];
end
for sub=1:NSUB  % over subjects
    ts=subject_task{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx2(sub)=1+lastupidx;
    nsubupidx2(sub)=nsublowidx2(sub)+size(ts,2)-1;
    lastupidx=nsubupidx2(sub);
    TS=[TS ts];
end
ts=TS;
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
Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

Isubdiag = find(tril(ones(LATDIM),-1));

for nsub=1:NSUB
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx(nsub)-nsublowidx(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx(nsub)+(np-1)*tpoints:nsublowidx(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataR{nsub}=patterns;
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx2(nsub)-nsublowidx2(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx2(nsub)+(np-1)*tpoints:nsublowidx2(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataT{nsub}=patterns;
end

%%%  Classification

cl=1:2;
pc=zeros(2,2);
for nfold=1:kfold
   shuffling=randperm(NSUB);
    TrainData1=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataR{sub};
        TrainData1=vertcat(TrainData1,TS);
    end
    XValidation1=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataR{sub};
        XValidation1=vertcat(XValidation1,TS);
    end
    Responses1=categorical(ones(size(TrainData1,1),1),cl);
    YValidation1=categorical(ones(size(XValidation1,1),1),cl);

    TrainData2=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataT{sub};
        TrainData2=vertcat(TrainData2,TS);
    end
    XValidation2=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataT{sub};
        XValidation2=vertcat(XValidation2,TS);
    end
    Responses2=categorical(2*ones(size(TrainData2,1),1),cl);
    YValidation2=categorical(2*ones(size(XValidation2,1),1),cl);

    TrainData=vertcat(TrainData1,TrainData2);
    XValidation=vertcat(XValidation1,XValidation2);
    Responses=vertcat(Responses1,Responses2);
    YValidation=vertcat(YValidation1,YValidation2);

    t = templateSVM('KernelFunction','rbf');
    svmmodel=fitcecoc(TrainData,Responses,'Learners',t);

    %% compute
    valno1=size(XValidation1,1);
    con=zeros(2,2);
    test1=predict(svmmodel,XValidation1);
    for i=1:valno1
        winclass=test1(i);
        con(1,winclass)=con(1,winclass)+1;
    end
    valno2=size(XValidation2,1);
    test2=predict(svmmodel,XValidation2);
    for i=1:valno2
        winclass=test2(i);
        con(2,winclass)=con(2,winclass)+1;
    end
    con(1,:)=con(1,:)/valno1;
    con(2,:)=con(2,:)/valno2;
    accdist(nfold)=sum(diag(con))/2;
    pc=pc+con;
end
pc=pc/kfold
acc=sum(diag(pc))/2

%% Classic Diff Map
Thorizont=3;
epsilon=300;

TS=[];
lastupidx=0;
for sub=1:NSUB  % over subjects
    ts=subject_rest{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx(sub)=1+lastupidx;
    nsubupidx(sub)=nsublowidx(sub)+size(ts,2)-1;
    lastupidx=nsubupidx(sub);
    TS=[TS ts];
end
for sub=1:NSUB  % over subjects
    ts=subject_task{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx2(sub)=1+lastupidx;
    nsubupidx2(sub)=nsublowidx2(sub)+size(ts,2)-1;
    lastupidx=nsubupidx2(sub);
    TS=[TS ts];
end
ts=TS;
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
Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

Isubdiag = find(tril(ones(LATDIM),-1));


for nsub=1:NSUB
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx(nsub)-nsublowidx(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx(nsub)+(np-1)*tpoints:nsublowidx(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataR{nsub}=patterns;
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx2(nsub)-nsublowidx2(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx2(nsub)+(np-1)*tpoints:nsublowidx2(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataT{nsub}=patterns;
end

%%%  Classification

cl=1:2;
pc=zeros(2,2);
for nfold=1:kfold
   shuffling=randperm(NSUB);
    TrainData1=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataR{sub};
        TrainData1=vertcat(TrainData1,TS);
    end
    XValidation1=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataR{sub};
        XValidation1=vertcat(XValidation1,TS);
    end
    Responses1=categorical(ones(size(TrainData1,1),1),cl);
    YValidation1=categorical(ones(size(XValidation1,1),1),cl);

    TrainData2=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataT{sub};
        TrainData2=vertcat(TrainData2,TS);
    end
    XValidation2=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataT{sub};
        XValidation2=vertcat(XValidation2,TS);
    end
    Responses2=categorical(2*ones(size(TrainData2,1),1),cl);
    YValidation2=categorical(2*ones(size(XValidation2,1),1),cl);

    TrainData=vertcat(TrainData1,TrainData2);
    XValidation=vertcat(XValidation1,XValidation2);
    Responses=vertcat(Responses1,Responses2);
    YValidation=vertcat(YValidation1,YValidation2);

    t = templateSVM('KernelFunction','rbf');
    svmmodel=fitcecoc(TrainData,Responses,'Learners',t);

    %% compute
    valno1=size(XValidation1,1);
    con=zeros(2,2);
    test1=predict(svmmodel,XValidation1);
    for i=1:valno1
        winclass=test1(i);
        con(1,winclass)=con(1,winclass)+1;
    end
    valno2=size(XValidation2,1);
    test2=predict(svmmodel,XValidation2);
    for i=1:valno2
        winclass=test2(i);
        con(2,winclass)=con(2,winclass)+1;
    end
    con(1,:)=con(1,:)/valno1;
    con(2,:)=con(2,:)/valno2;
    accdistQ(nfold)=sum(diag(con))/2;
    pc=pc+con;
end
pcQ=pc/kfold
accQ=sum(diag(pcQ))/2

%% PCA

TS=[];
lastupidx=0;
for sub=1:NSUB  % over subjects
    ts=subject_rest{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx(sub)=1+lastupidx;
    nsubupidx(sub)=nsublowidx(sub)+size(ts,2)-1;
    lastupidx=nsubupidx(sub);
    TS=[TS ts];
end
for sub=1:NSUB  % over subjects
    ts=subject_task{sub}.dbs80ts;
    ts=ts(indexregion,1:Tmax);
    clear signal_filt;
    for seed=1:N
        ts(seed,:)=detrend(ts(seed,:)-nanmean(ts(seed,:)));
        signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
    end
    ts=signal_filt(:,10:end-10);
    ts=zscore(ts,[],2);
    nsublowidx2(sub)=1+lastupidx;
    nsubupidx2(sub)=nsublowidx2(sub)+size(ts,2)-1;
    lastupidx=nsubupidx2(sub);
    TS=[TS ts];
end
ts=TS;
[~,PhiPCA,llpca]=pca(TS');

Phi=PhiPCA(:,1:LATDIM)*diag(llpca(1:LATDIM));
Isubdiag = find(tril(ones(LATDIM),-1));

for nsub=1:NSUB
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx(nsub)-nsublowidx(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx(nsub)+(np-1)*tpoints:nsublowidx(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataR{nsub}=patterns;
    if FULLWIN==1
        Npatterns=1;
        WIN=nsubupidx2(nsub)-nsublowidx2(nsub);
        tpoints=0;
    else
        tpoints=2;Npatterns=floor((nsubupidx(nsub)-nsublowidx(nsub)-WIN)/2);
    end
    clear patterns;
    for np=1:Npatterns
        FC=corrcoef(Phi(nsublowidx2(nsub)+(np-1)*tpoints:nsublowidx2(nsub)+(np-1)*tpoints+WIN,:));
        patterns(np,:)=FC(Isubdiag);
    end
    DataT{nsub}=patterns;
end

%%%  Classification

cl=1:2;
pc=zeros(2,2);
for nfold=1:kfold
   shuffling=randperm(NSUB);
    TrainData1=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataR{sub};
        TrainData1=vertcat(TrainData1,TS);
    end
    XValidation1=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataR{sub};
        XValidation1=vertcat(XValidation1,TS);
    end
    Responses1=categorical(ones(size(TrainData1,1),1),cl);
    YValidation1=categorical(ones(size(XValidation1,1),1),cl);

    TrainData2=[];
    for sub=shuffling(1:NTRAIN)
        TS=DataT{sub};
        TrainData2=vertcat(TrainData2,TS);
    end
    XValidation2=[];
    for sub=shuffling(NTRAIN+1:end)
        TS=DataT{sub};
        XValidation2=vertcat(XValidation2,TS);
    end
    Responses2=categorical(2*ones(size(TrainData2,1),1),cl);
    YValidation2=categorical(2*ones(size(XValidation2,1),1),cl);

    TrainData=vertcat(TrainData1,TrainData2);
    XValidation=vertcat(XValidation1,XValidation2);
    Responses=vertcat(Responses1,Responses2);
    YValidation=vertcat(YValidation1,YValidation2);

    t = templateSVM('KernelFunction','rbf');
    svmmodel=fitcecoc(TrainData,Responses,'Learners',t);

    %% compute
    valno1=size(XValidation1,1);
    con=zeros(2,2);
    test1=predict(svmmodel,XValidation1);
    for i=1:valno1
        winclass=test1(i);
        con(1,winclass)=con(1,winclass)+1;
    end
    valno2=size(XValidation2,1);
    test2=predict(svmmodel,XValidation2);
    for i=1:valno2
        winclass=test2(i);
        con(2,winclass)=con(2,winclass)+1;
    end
    con(1,:)=con(1,:)/valno1;
    con(2,:)=con(2,:)/valno2;
    accdistPCA(nfold)=sum(diag(con))/2;
    pc=pc+con;
end
pcPCA=pc/kfold
accPCA=sum(diag(pcPCA))/2

for i=1:100
    accdist2(i)=mean(accdist(1+(i-1)*100:(i-1)*100+100));
    accdistQ2(i)=mean(accdistQ(1+(i-1)*100:(i-1)*100+100));
end

a=accdist2;
b=accdistQ2;
stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],1000,0.01,'ranksum');
ppGQ=min(stats.pvals)

save results_class_HCPsocial_concatenated.mat acc accQ accPCA accdist accdistQ pc pcQ pcPCA;