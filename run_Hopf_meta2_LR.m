clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));

NSUB=100;
LRange=50;
N=62;
LATDIM=7;

Isubdiag = find(tril(ones(N),-1));
IsubdiagL = find(tril(ones(LATDIM),-1));

indexregion=[1:31 50:80];

load results_f_diff_REST_dk62.mat;
load SC_dbs80HARDIFULL.mat;
C = SC_dbs80HARDI;
C = C/max(max(C));
C=C(indexregion,indexregion);

TR=0.72;  % Repetition Time (seconds)
Tmax = 1200;
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

sig = 0.01;
dt=0.1*TR/2;
dsig = sqrt(dt)*sig;

AA=-0.02;

a=AA*ones(N,1);
a=repmat(a,1,2);
wo = f_diff'*(2*pi);
omega = repmat(wo,1,2);
omega(:,1) = -omega(:,1);

%%  SC
load SC_dbs80HARDIFULL.mat;
C = SC_dbs80HARDI;
C = C/max(max(C));
C=C(indexregion,indexregion);

%% WITHOUT LR
load cog_dk80.mat;
dk80cog=dk80cog(indexregion,:);
for i=1:N
    for j=1:N
        SCdist(i,j)=sqrt(sum((dk80cog(i,:)-dk80cog(j,:)).^2));
        if SCdist(i,j)>LRange
            SCmasklr(i,j)=0;
        else
            SCmasklr(i,j)=1;
        end
    end
end
C_noLR=C.*SCmasklr;

IsubdiagT = find(tril(ones(Tmax),-1));

%% FULL WITH LR

for trial=1:50
    trial
    for sub=1:NSUB
        G=0.2;
        G2=G-0.005+0.01*rand;
        wC = G2*C;
        sumC = repmat(sum(wC,2),1,2);
        clear EdgesA Edges;
        xs=zeros(Tmax,N);
        z = 0.1*ones(N,2);
        nn=0;
        % discard first 2000 time steps
        for t=0:dt:1000
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
        end
        % actual modeling (x=BOLD signal (Interpretation), y some other oscillation)
        for t=0:dt:((Tmax-1)*TR)
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=z(:,1)';
            end
        end
        ts=xs';
        for seed=1:N
            ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:)));
            signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
        end

        %% Edges
        zPhi=zscore(signal_filt');
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesA(:,t)=fcd(Isubdiag)';
        end
        FCDA=(EdgesA'*EdgesA)./(vecnorm(EdgesA)'*vecnorm(EdgesA));
        MetaA(sub)=0.5*(log(2*pi*var(FCDA(IsubdiagT))))+0.5;

        %% Meta Diff
        epsilon=400;
        ts1=signal_filt;
        ts=zscore(ts1,[],2);
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
        Phi=VV(:,1:LATDIM);
        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));
        zPhi=zscore(Phi);

        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesL(:,t)=fcd(IsubdiagL)';
        end

        FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
        Meta2(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

        %% Meta Q
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
        Phi=VV(:,1:LATDIM);
        Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));
        zPhi=zscore(Phi);
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesL(:,t)=fcd(IsubdiagL)';
        end

        FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
        MetaQ2(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

        %% Without LR
 
        G=0.4;
        G2=G-0.005+0.01*rand;
        wC = G2*C; 
        sumC = repmat(sum(wC,2),1,2);
        clear EdgesA Edges;
        xs=zeros(Tmax,N);
        z = 0.1*ones(N,2);
        nn=0;
        % discard first 2000 time steps
        for t=0:dt:1000
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
        end
        % actual modeling (x=BOLD signal (Interpretation), y some other oscillation)
        for t=0:dt:((Tmax-1)*TR)
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=z(:,1)';
            end
        end
        ts=xs';
        for seed=1:N
            ts(seed,:)=detrend(ts(seed,:)-mean(ts(seed,:)));
            signal_filt(seed,:) =filtfilt(bfilt,afilt,ts(seed,:));
        end

        %% Edges
        zPhi=zscore(signal_filt');
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesA(:,t)=fcd(Isubdiag)';
        end
        FCDA=(EdgesA'*EdgesA)./(vecnorm(EdgesA)'*vecnorm(EdgesA));
        MetaA_noLR(sub)=0.5*(log(2*pi*var(FCDA(IsubdiagT))))+0.5;

        %% Meta Diff
        epsilon=400;
        ts1=signal_filt;
        ts=zscore(ts1,[],2);
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
        Phi=VV(:,1:LATDIM);
        Phi=Phi*(LL(2:LATDIM+1,2:LATDIM+1));
        zPhi=zscore(Phi);

        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesL(:,t)=fcd(IsubdiagL)';
        end

        FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
        Meta_noLR2(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

        %% Meta Q
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
        Phi=VV(:,1:LATDIM);
        Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));
        zPhi=zscore(Phi);
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            EdgesL(:,t)=fcd(IsubdiagL)';
        end

        FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
        MetaQ_noLR2(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;
    end
    Meta(trial)=corr2(MetaA,Meta2);
    MetaQ(trial)=corr2(MetaA,MetaQ2);
    Meta_noLR(trial)=corr2(MetaA_noLR,Meta_noLR2);
    MetaQ_noLR(trial)=corr2(MetaA_noLR,MetaQ_noLR2);
end

figure(1)
subplot(1,2,1)
violinplot([Meta' Meta_noLR']);
ylim([-0.3 0.6])
subplot(1,2,2)
violinplot([MetaQ' MetaQ_noLR']);
ylim([-0.3 0.6])

diffMeta=Meta-Meta_noLR;
diffMetaQ=MetaQ-MetaQ_noLR;

figure(2)
violinplot([diffMeta' diffMetaQ'])
ylim([-0.4 0.7])
axis('square');
ranksum(diffMeta,diffMetaQ)

diffMetas=MetaQ-Meta;
diffMetas_noLR=MetaQ_noLR-Meta_noLR;

figure(3)
violinplot([diffMetas' diffMetas_noLR'])
ylim([-0.4 0.7])
axis('square');
ranksum(diffMetas,diffMetas_noLR)


save results_hopf_LR.mat Meta MetaQ Meta_noLR MetaQ_noLR;