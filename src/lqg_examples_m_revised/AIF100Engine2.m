%%%%%%%%%%%%%%%  F-100 Engine %%%%%%%%%%%%%

%%
A=[-3.91800e+00 4.1886e+00 -4.1148e-02 1.2279e-01;
-1.8061e-01 -2.1480e+00 1.5853e-01 6.6994e-04;
-1.3190e-01 -2.4056e-01 -6.6630e-01 2.37700e-04;
-3.8191e-01 -1.0501e+00 -6.7400e-02 -2.0000e+00]



B=[
5.1991e-01 1.1942e+00 2.1974e-01 -2.4990e-02 -1.7226e-02;
 3.6266e-01 1.0836e-01 7.2562e-03 -1.2133e-02 -7.2114e-03; 
2.84270e-01 3.3231e-02 5.7770e-03 5.7672e-03 1.6319e-03;
 9.3743e-01 7.3072e-02 1.7417e-02 2.0418e-02 1.0634e-01]

C=[ 2.2043e+01 0.0000e+00 0.0000e+00 0.0000e+00;
  0.0000e+00 2.7339e+01 0.0000e+00 0.0000e+00;
 3.7700e+00 1.0341e+01 -7.6298e-03 -4.3237e-03;
  8.0543e+00 3.1436e-01 -6.6634e-02 -3.7135e-02;
-2.9070e+00 -7.9884e+00 -5.1265e-01 2.6855e-03] 

D=[0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00;
0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 .0000e+00;
1.0036e+00 -8.2350e-01 -1.5200e-01 -5.6233e-02 -5.8600e-02;
9.7674e-01 -5.7450e+00 -3.8500e-01 9.5762e-03 -2.2963e-02;
 7.1316e+00 5.5560e-01 1.3247e-01 1.5533e-01 4.8290e-02]

Q=C'*C;
R=eye(5);
[K,S,ECLP]=lqr(A,B,Q,R)



[n,m]=size(B);
[p,n]=size(C);

%% Klaman filter
Qe=eye(m);
Re=0.01*eye(m);
[L,P,E] = lqe(A,B,C,Qe,Re);

%% LQ with Feedforward 

NN=[A B;
    C D];
NN=inv(NN)*[zeros(n,m);eye(m)];
Nx=NN(1:n,:);
Nu=NN(n+1:n+m,:);
Nbar=Nu+K*Nx;

%% Step response
%t=0:.01:10;
figure(1)
step(A-B*K,B*Nbar,C-D*K,D*Nbar);
grid on;

%% LQG: With the Kalman Filter

Acl=[A -B*K;
    L*C A-B*K-L*C];
Bcl=[B*Nbar;
    B*Nbar];
Ccl=[C -D*K];
Dcl=D*Nbar;


figure(1)
step(Acl,Bcl,Ccl,Dcl)
grid on;

%% Integral Control

AA=[zeros(p,p) C;
    zeros(n,p) A];
BB=[D;
    B];
QQ=eye(n+m);
RR=eye(m);
[KK,SS,EECLP]=lqr(AA,BB,QQ,RR);
K1=KK(:,1:m);
Ko=KK(:,m+1:n+m);

Ac=zeros(p,p);
Bc=-K1;
Cc=eye(m);
Dc=-Ko;

%%%% CL DC GAIN OF IDENTITY IS RIGHT!


AAcl=[A B*Cc B*Dc;
      Bc*C Ac+Bc*D*Cc  Bc*D*Dc;
      L*C B*Cc A-L*C+B*Dc];
BBcl=[zeros(n,m);
      -Bc;
      zeros(n,m)];
CCcl=[C  D*Cc D*Dc];
DDcl=zeros(p,m);


figure(3)
step(AAcl,BBcl,CCcl,DDcl)
grid on;

%%%
SYS=ss(AAcl,BBcl,CCcl,DDcl);

%% Sensitivity Function
figure(4)
SSYS=eye(m)-SYS;
w=logspace(-3,3);
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

PM = max(PM1,PM2)
