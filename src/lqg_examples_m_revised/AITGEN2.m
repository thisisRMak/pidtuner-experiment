%%%%%%%%%% TGEN  Turbo-generator Maciejowski  %%%%%%%%%
A=[-18.4456 4.2263 -2.2830 0.2260 0.4220 -0.0951;
    -4.0977 -6.0706 5.6825 -0.6966 -1.2246 0.2873;
    1.4449 1.4336 -2.6477 0.6092 0.8979 -0.2300;
    -0.00093 0.232 -0.5002 -0.1764 -6.3152 0.1350;
    -0.0464 -0.3489 0.7238 6.3117 -0.6886 0.3645;
    -0.0602 -0.2361 0.2300 0.0915 -0.3214 -0.2087];
B=[-0.2748 3.1463;
    -0.0501 -9.3737;
    -0.1550 7.4296;
    -0.0716 -4.9176;
    -0.0814 -10.2648;
    0.0244 13.7943];
C=[0.5971 -0.7697 4.8850 4.8608 -9.8177 -8.8610;
    3.1013 9.3422 -5.6000 -0.7490 2.9974 10.5719];
D=[0 0;
    0 0];

Q=C'*C;
R=eye(2);
[K,S,ECLP] = lqr(A,B,Q,R)


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
RR=0.001*eye(m);
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

PM=max(PM1,PM2)

