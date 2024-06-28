clear all;
path2=[ '../../Tenet/TENET/'];
addpath(genpath(path2));
path3=[ '../../Turbulence/Basics/'];
addpath(genpath(path3));

N=62;
Isubdiag = find(tril(ones(N),-1));

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
G_range=0.2:0.001:0.3; %%0.15:0.01:0.7;
for G=G_range
    G
    wC = G*C;
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
        FCsim=corrcoef(ts(:,50:1150)');
        fitt2(sub)=corr2(FCsim(Isubdiag),FCemp(Isubdiag));
        err2(sub)=mean((FCemp(Isubdiag)-FCsim(Isubdiag)).^2);

        %% Edges
        zPhi=zscore(signal_filt');
        for t=1:size(zPhi,1)
            fcd=zPhi(t,:)'*zPhi(t,:);
            Edges(:,t)=fcd(Isubdiag)';
        end
        FCD=dist(Edges);
        Metastability2(sub)=0.5*(log(2*pi*var(FCD(:))))+0.5;

        %% Kuramoto

        KoP=abs(nansum(complex(cos(Phases),sin(Phases)),1))/N;
        KoP=KoP(20:end-20);
        KuramotoOrderParameter2(sub)=std(KoP);

    end
    fitt(ng)=mean(fitt2);
    err(ng)=mean(err2);
    fitts(ng)=std(fitt2);
    errs(ng)=std(err2);
    Metastability(ng)=mean(Metastability2);
    Metastabilitys(ng)=std(Metastability2);
    KuramotoOrderParameter(ng)=mean(KuramotoOrderParameter2)
    KuramotoOrderParameters(ng)=std(KuramotoOrderParameter2)
    ng=ng+1;
end

shadedErrorBar(G_range,fitt,fitts,'k',0.7);
hold on;
shadedErrorBar(G_range,err,errs,'r',0.7);
shadedErrorBar(G_range,Metastability,Metastabilitys,'c',0.7);
shadedErrorBar(G_range,KuramotoOrderParameter,KuramotoOrderParameters,'b',0.7);

save results_hopf_fitt_KoP_fineG0203.mat fitt fitts err errs KuramotoOrderParameter KuramotoOrderParameters ...
    Metastability Metastabilitys G_range;