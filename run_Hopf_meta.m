clear all;
path2=[ '../../Tenet/TENET/'];
addpath(genpath(path2));
path3=[ '../../Turbulence/Basics/'];
addpath(genpath(path3));

N=62;
LATDIM=7;

Isubdiag = find(tril(ones(N),-1));
IsubdiagL = find(tril(ones(LATDIM),-1));

index=[1:31 50:80];

load results_f_diff_REST_dk62.mat;
load SC_dbs80HARDIFULL.mat;
C = SC_dbs80HARDI;
C = C/max(max(C));
C=C(index,index);

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

ng=1;
G_range=0.05:0.1:0.65;
for G=G_range
    nnG=1;
    for G2=G-0.05:0.001:G+0.05
        G2
        wC = G2*C;
        sumC = repmat(sum(wC,2),1,2);
        for sub=1:30
            xs=zeros(Tmax,N);
            z = 0.1*ones(N,2);
            nn=0;
            % discard first 2000 time steps
            for t=0:dt:2000
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
                Xanalytic = hilbert(demean(signal_filt(seed,:)));
                Phases(seed,:) = angle(Xanalytic);
            end

            KuraMeta2(sub)=std(abs(sum(exp(1i*Phases),1))/N);

            %% Edges
            zPhi=zscore(signal_filt');
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                EdgesA(:,t)=fcd(Isubdiag)';
            end
            FCD=dist(EdgesA);
            Metastability2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;

            %% Meta Diff

            %% second simulation
            xs=zeros(Tmax,N);
            z = 0.1*ones(N,2);
            nn=0;
            % discard first 2000 time steps
            for t=0:dt:2000
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
            %%

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
            Phi=VV(:,2:LATDIM+1);
            Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

            zPhi=zscore(Phi);
            for t=1:Tm
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edges(:,t)=fcd(IsubdiagL)';
            end

            FCD=dist(Edges);
            Meta2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;

            %% Meta Q
            %% second simulation
            xs=zeros(Tmax,N);
            z = 0.1*ones(N,2);
            nn=0;
            % discard first 2000 time steps
            for t=0:dt:2000
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

            epsilon=200;
            Thorizont=3;
            ts1=signal_filt;
            ts=zscore(ts1,[],2);
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

            zPhi=zscore(Phi);
            for t=1:Tm
                fcd=zPhi(t,:)'*zPhi(t,:);
                Edges(:,t)=fcd(IsubdiagL)';
            end

            FCD=dist(Edges);
            MetaQ2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;
        end
        KuraMeta(nnG)=mean(KuraMeta2);
        Metastability(nnG)=mean(Metastability2);
        Meta(nnG)=mean(Meta2);
        MetaQ(nnG)=mean(MetaQ2);
        nnG=nnG+1;
    end
    corr_metaC(ng)=corr2(Metastability,Meta)
    corr_metaQ(ng)=corr2(Metastability,MetaQ)
    mean_Metastability(ng)=mean(Metastability);
    mean_metaC(ng)=mean(Meta);
    mean_metaQ(ng)=mean(MetaQ);
    KuramotoMetastability(ng)=mean(KuraMeta)
    mean_Metastabilitys(ng)=std(Metastability);
    mean_metaCs(ng)=std(Meta);
    mean_metaQs(ng)=std(MetaQ);
    ng=ng+1;
end

figure(1)
plot(G_range,corr_metaC,'k');
hold on;
plot(G_range,corr_metaQ,'r');

figure(2)
plot(G_range,KuramotoMetastability);

figure(3)
shadedErrorBar(G_range,mean_Metastability,mean_Metastabilitys,'b',0.7);
hold on;
shadedErrorBar(G_range,mean_metaC,mean_metaCs,'k',0.7);
shadedErrorBar(G_range,mean_metaQ,mean_metaQs,'r',0.7);

save results_hopf_meta.mat corr_metaC corr_metaQ mean_Metastability mean_metaC mean_metaQ ...
    mean_Metastabilitys mean_metaCs mean_metaQs KuramotoMetastability G_range;