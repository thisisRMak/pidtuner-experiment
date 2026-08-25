%%%%%%%%%%%%%%%% RPV Maciejowski     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%
A=[-0.02567 -36.617 -18.897 -32.090 3.2509 -0.76257;
    9.257e-5 -1.8977 0.98312 -7.256e-4 -0.1708 -4.965e-3;
    0.012338 11.720 -2.6316 8.758e-4 -31.604 22.396;
    0 0 1 0 0 0;
    0 0 0 0 -30 0;
    0 0 0 0 0 -30];
B=[0 0;
    0 0;
    0 0;
    0 0;
    30 0;
    0 30];
C=[0 1 0 0 0 0;
    0 0 0 1 0 0];
D=[ 0 0;
    0 0];

Q=C'*C;
R=eye(2);
[K,S,CLP] = lqr(A,B,Q,R)

[n,m]=size(B);
[p,n]=size(C);

%% Klaman filter
Qe=eye(m);
Re=eye(m);
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
step(A-B*K,B*Nbar,C,D);
grid on;

%% LQG: With the Kalman Filter

Acl=[A -B*K;
    L*C A-B*K-L*C];
Bcl=[B*Nbar;
    B*Nbar];
Ccl=[C zeros(p,n)];
Dcl=D;

figure(2)
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

Bc=-K1;
Cc=eye(m);

AAcl=[A B*Cc -B*Ko;
      Bc*C zeros(m,m)  zeros(m,n);
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

PM=max(PM1,PM2)

