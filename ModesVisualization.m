clear all;
path2=[ '../Turbulence/Basics'];
addpath(path2);
path3=[ '../Nonequilibrium/'];
addpath(genpath(path3));
path4=[ '../Tenet/TENET/'];
addpath(genpath(path4));
path5=[ '../LaplaceManifold/Sleep/'];
addpath(genpath(path5));

NSUB=1003
N=62;

LATDIM=7;


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

load('db80Yeo7_parcial_belonging.mat');
yeo=Yeo_in_new_Parcel(:,indexregion);
load('hcp1003_REST1_LR_dbs80.mat');

versor=eye(LATDIM);

%%%%%% Quantum
epsilon=300;
Thorizont=3;

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

    %%  find RSN label
    for t=1:Tm
        for red=1:7
            overlapred(t,red)=corr2(ts(:,t)',yeo(red,:));
        end
    end
    %%
    ts=zscore(ts,[],2);
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

    Phi=zscore(Phi);
    for t=1:Tm
        [aux,indexlat(t)]=max(abs(Phi(t,:)));
    end

    for bm=1:LATDIM
        yeobmQ2(bm,:)=mean(overlapred(indexlat==bm,:));
        MM=max(yeobmQ2(bm,:));
        mm=min(yeobmQ2(bm,:));
        yeobmQ(sub,bm,:)= 2/(MM-mm)*yeobmQ2(bm,:)-2*mm/(MM-mm)-1;
    end

end


for s1=1:NSUB
    for s2=1:NSUB
        sub1=squeeze(yeobmQ(s1,:,:));
        sub2=squeeze(yeobmQ(s2,:,:));
        for i=1:7
            for j=1:7
                dd(i,j)=sum((sub1(i,:).*sub2(j,:)))/norm(sub1(i,:))/norm(sub2(j,:));
            end
        end
        ddsub(s1,s2)=max(mean(max(dd)),mean(max(dd')));
        [aux ii]=max([mean(max(dd)),mean(max(dd'))]);
        if ii==1
            [aux ind]=max(dd);
        else
            [aux ind]=max(dd');
        end
        unisub(s1,s2)=length(ind)-length(unique(ind));
    end
end

Isubdiags = find(tril(ones(NSUB),-1));

topdd=ddsub(Isubdiags);
topunisub=unisub(Isubdiags);

ii1=find(topdd>=quantile(topdd,0.999));
ii2=find(topunisub<=quantile(topunisub,0.001));

[ss1 ss2]=ind2sub(NSUB,intersect(ii1,ii2));
bestsub=unique(union(ss1,ss2));

save results_modes_visualization.mat bestsub yeobmQ ss1 ss2 unisub ddsub topdd topunisub;
