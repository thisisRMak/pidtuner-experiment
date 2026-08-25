%%%%%%%%%%%%% Aircraft Model  Hall     %%%%%%%%
A=[-2.97e-2 -1 0 4.38e-2 0;
    3.31e-1 -4.2e-3 -4.61e-2 0 0;
    -1.13 1.28e-1 -8.03e-1 0 0;
    0 0 1 0 0;
    0 1 0 0 0]
B=[0 0;
    3.81e-1 4.0e-2;
    6.7e-2 1.59;
    0 0;
    0 0]
C=[0 0 0 1 0;
    0 0 0 0 1]
D=[0 0;
    0 0]
Q=[1 0 0 0 1;
    0 0 0 0 0;
    0 0 0 0 0;
    0 0 0 1 0;
    1 0 0 0 1];
R=[1 0;
    0 1];
[K,S,ECLP]=lqr(A,B,Q,R)

[n,m]=size(B);
[p,n]=size(C);

%% Klaman filter
Qe=eye(2);
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



figure(1)
contourf(SS)
title('Riccati Matrix');
colorbar

figure(2)
contourf(KK)
title('Feedback Gain Matrix');
colorbar

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
w=logspace(-1,3);
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
