%%%%%%%% Distillation Column  Davison 2011 %%%%%%%
%%%%%
A=1e-2*[-1.4 0.43 0 0 0 0 0 0 0 0 0;
    0.95 -1.38 0.46 0 0 0 0 0 0 0 0.05;
    0 0.95 -1.41 0.63 0 0 0 0 0 0 0.02;
    0 0 0.95 -1.58 1.1 0 0 0 0 0 0;
    0 0 0 0.95 -3.12 1.50 0 0 0 0 0;
    0 0 0 0 2.02 -3.52 2.20 0 0 0 0;
    0 0 0 0 0 2.02 -4.22 2.80 0 0 0;
    0 0 0 0 0 0 2.02 -4.82 3.70 0 0.02;
    0 0 0 0 0 0 0 2.02 -5.72 4.20 0.05;
    0 0 0 0 0 -0 0 0 2.02 -4.83 0.05;
    2.55 0 0 0 0 0 0 0 0 2.55 -1.85];

B=1e-4*[0 0;
    0.05 25;
    0.02 50;
    0.01 50;
    0 50;
    0 50;
    -0.05 50;
    -0.10 50;
    -0.4 25;
    -0.2 25;
    4.6 0];
%C=[0 0 0 0 0 0 0 0 0 1 0;
%   1 0 0 0 0 0 0 0 0 0 0;
%    0 0 0 0 0 0 0 0 0 0 1];
%D=[0 0;
%    0 0;
%    0 0];
C=[0 0 0 0 0 0 0 0 0 1 0;
   1 0 0 0 0 0 0 0 0 0 0];
D=[0 0;
   0 0];


[n,m]=size(B);
[p,n]=size(C);

Q=eye(n);
R=0.1*eye(m);
[K,S,CLP]=lqr(A,B,Q,R)

[n,m]=size(B);
[p,n]=size(C);

%% Klaman filter
Qe=eye(2);
Re=0.01*eye(p);
[L,P,E] = lqe(A,B,C,Qe,Re);





%% Integral Control

AA=[zeros(p,p) C;
    zeros(n,p) A];
BB=[D;
    B];
QQ=eye(n+p);
RR=eye(m);
[KK,SS,EECLP]=lqr(AA,BB,QQ,RR);
K1=KK(:,1:p);
Ko=KK(:,p+1:n+p);

Bc=-K1;
Cc=[eye(p)];

Ac=zeros(p,p);

figure(1)
contourf(SS)
title('Riccati Matrix');
colorbar

figure(2)
contourf(KK)
title('Feedback Gain Matrix');
colorbar


AAcl=[A B*Cc -B*Ko;
      Bc*C Ac  zeros(p,n);
      L*C B*Cc A-B*Ko-L*C];
BBcl=[zeros(n,m);
      -Bc;
      zeros(n,m)];
CCcl=[C zeros(p,p) zeros(p,n)];
DDcl=zeros(p,m);


figure(3)
step(AAcl,BBcl,CCcl,DDcl)
grid on;

%%%
SYS=ss(AAcl,BBcl,CCcl,DDcl);

%% Sensitivity Function
figure(4)
SSYS=eye(m)-SYS;
w=logspace(-4,3);
[SSV,w]=sigma(SSYS,w);
loglog(w,SSV(1,:),'LineWidth',2)
xlabel('\omega (rad/sec)');
ylabel('||S||')
title('Sensitivity Function')
grid on;

%% Complementary Sensitivity Function

figure(5)
SYS=ss(AAcl,BBcl,CCcl,DDcl);
[SSVV,w]=sigma(SYS,w);
loglog(w,SSVV(1,:),'LineWidth',2)
xlabel('\omega (rad/sec)');
ylabel('||T||')
title('Complementary Sensitivity Function');
grid on;












  %%% Phase margin based on S

alpha=max(SSV(1,:));
alpha=1/alpha;
PM1=2*asin(alpha/2);
ff=180/pi;
% Phase margin in degrees

PM1=PM1*ff

%%% Gain Margin based on S

GM1=1/(1-alpha);
% GM in db
%GM1=20*log10(GM1)

%%% Phase margin based on T

beta=max(SSVV(1,:));
beta=1/beta;
PM2=2*asin(beta/2);
ff=180/pi;
% Phase margin in degrees

PM2=PM2*ff

%%% Gain Margin based on T

GM2=1+beta;
% GM in db
%GM1=20*log10(GM2)


GMplus = max(1+alpha,1/(1-beta))

GMminus = max(1-alpha,1/(1+beta))

PM=max(PM1,PM2)
