clear all;
path2=[ '../../Tenet/TENET/'];
addpath(genpath(path2));
path3=[ '../../Turbulence/Basics/'];
addpath(genpath(path3));

N=62;
LATDIM=7;
NSUB=100;

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
IsubdiagT = find(tril(ones(Tmax),-1));

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
G_range=0.025:0.025:0.7;
for G=G_range
    G
    for trial=1:50
        for sub=1:NSUB
            G2=G-0.005+0.01*rand;
            wC = G2*C;
            sumC = repmat(sum(wC,2),1,2);
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
            Phi=VV(:,2:LATDIM+1);
            Phi=Phi*abs(LL(2:LATDIM+1,2:LATDIM+1));

            zPhi=zscore(Phi);
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                EdgesL(:,t)=fcd(IsubdiagL)';
            end

            FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
            Meta(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

            %% Meta Q
            epsilon=300;
            Thorizont=2;
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
            for t=1:size(zPhi,1)
                fcd=zPhi(t,:)'*zPhi(t,:);
                EdgesL(:,t)=fcd(IsubdiagL)';
            end

            FCD=(EdgesL'*EdgesL)./(vecnorm(EdgesL)'*vecnorm(EdgesL));
            MetaQ(sub)=0.5*(log(2*pi*var(FCD(IsubdiagT))))+0.5;

        end
        corrMeta2(trial)=corr2(MetaA,Meta);
        corrMetaQ2(trial)=corr2(MetaA,MetaQ);
    end
    corrMeta(ng)=mean(corrMeta2)
    corrMetaQ(ng)=mean(corrMetaQ2)
    corrMetas(ng)=std(corrMeta2);
    corrMetaQs(ng)=std(corrMetaQ2);
    ng=ng+1;
end


figure(1)
shadedErrorBar(G_range,corrMeta,corrMetas,'k',0.7);
hold on;
shadedErrorBar(G_range,corrMetaQ,corrMetaQs,'r',0.7);
axis('square');

save results_hopf_meta_Th3.mat corrMeta corrMetaQ corrMetas corrMetaQs G_range;