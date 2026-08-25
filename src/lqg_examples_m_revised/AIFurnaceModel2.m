%%%%%%%%%%%%%%%% Furnace System  Davison 2011, Rosenbrock %%%%%%%
%%%%%
%%%%%
A=1e-2*[-23.226 1.712 1.1 0.774 0.093 1.456 -1.060 0.453;
    0.232 -25.493 0.318 -2.497 -2.235 -3.691 0.133 -1.226;
    0.666 0.531 -25.651 2.520 2.862 3.182 -0.386 0.568;
    2.531 -1.134 -2.152 -24.043 2.580 -1.283 -0.503 0.729;
    0.546 -1.779 -3.258 2.397 -20.030 0.938 0.350 -0.546;
    -2.242 3.054 3.165 1.670 -1.008 -18.773 -0.100 0.204;
    -5.724 -2.789 -1.336 -0.311 -0.211 0.546 -22.403 -0.030;
    -0.183 -1.271 -0.737 0.607 -0.525 -0.145 0.074 -20.380];
B=[-.334 -0.223 -0.4942 -0.416;
    -0.161 -0.247 0.1345 0.330;
    0.148 -0.329 0.0593 -0.435;
    0.199 -0.270 -0.2105 -0.258;
    -0.157 0.245 0.0557 0.281;
    0.076 -0.048 -0.0740 -0.024;
    -0.020 0.050 0.0288 0.009;
    -0.038 0.058 0.0084 0.062];
C=[-0.4177 -0.3583 -0.3344 0.3396 -0.2580 -0.0917 0.0403 -0.0567;
    -0.2647 -0.4481 -0.1575 -0.1221 -0.0002 0.0932 0.0006 0.0003;
    -0.3225 0.0137 0.3904 -0.2739 0.2818 0.0959 -0.0427 0.0692;
    -0.2685 0.2041 -0.2276 0.0027 -0.0905 0.0186 0.0195 -0.0247];
D=[0 0 0 0;
   0 0 0 0;
   0 0 0 0;
   0 0 0 0];
Q=eye(8);
R=0.1*eye(4);
[K,S,CLP]=lqr(A,B,Q,R);



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

