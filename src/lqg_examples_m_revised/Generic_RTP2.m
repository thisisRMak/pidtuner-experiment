%%%%%%%%%%%%%%%% Generic RTP   %%%%%%%%%%%%%%%%
load rtpsystem.dat
ny = 5;
nu = 5;
nx = 15;
dt = 0.000000;
A = rtpsystem(1:nx,1:nx);
B = rtpsystem(1:nx,nx+(1:nu));
C = rtpsystem(nx+(1:ny),1:nx);
D = rtpsystem(nx+(1:ny),nx+(1:nu));
%rtpsys = grp(a,b,c,d);
P = ss(A,B,C,D);
figure; bode(P)
E=eig(A);
figure()
plot(real(E),imag(E),'x');
grid on
Z=tzero(A,B,C,D);
ZZ=Z(6:end);
hold on;
plot(real(ZZ),imag(ZZ),'o');
title('Pole Transmission Zero Map');
figure()
step(P)
grid on
figure()
bode(P)
grid on;
[SysBal,G]=balreal(P);
figure()
semilogy(G/G(1));
grid on;
title('Normalized Hankel SVs');
[Abar,Bbar,Cbar,T,K]=ctrbf(A,B,C);
[Abar,Bbar,Cbar,T,K]=obsvf(A,B,C);
Wc=gram(P,'c')
figure()
surf(Wc)
title('Controllability Gramian');
Wo=gram(P,'o')
figure()
surf(Wo)
title('Observability Gramian');
% Input Output Directions
DC=dcgain(P)
[u,s,v]=svd(DC)

%%
sigma(P)
grid on
title('Singular Values of the Plant');

%%%%% Robust Servo Design %%%%%%
Ar=0*eye(5);
%%%%
As=[Ar C;
    zeros(15,5) A];
Bs=[D;
    B];
Am=diag([-5 -5 -5 -5 -5]);
%Q1=eye(5);
Q1=[5 0 0 0 0;
    0 5 -1 -1 -1;
    0 -1 5 -1 -1;
    0 -1 -1 5 -1;
    0 -1 -1 -1 5];
Q2=0.001*eye(5);

Cm=[ eye(5) zeros(5,15)];
Q1hat=(Cm*As-Am*Cm)'*Q1*(Cm*As-Am*Cm)
K=lqr(As,Bs,Q1hat,Q2)
eig(As-Bs*K)

K1=K(:,1:5);
Ko=K(:,6:20);
Ac=Ar;
Bc=-K1;
Cc=eye(5);
Dc=0*eye(5);

%%% r to y
Acl=[A-B*Ko B*Cc;
    Bc*C Ac];
Bcl=[zeros(15,5);-Bc];
Ccl=[C zeros(5,5)];
Dcl=[0*eye(5)];

%%% r to e
Ccle=[C zeros(5,5)];
Dcle=-eye(5);
%%%% blocking zeros
tzero(Acl,Bcl,Ccle,Dcle)

t=0:.01:20;

SYS=ss(Acl,Bcl,Ccl,Dcl);

figure()
step(SYS,t);

ylabel('r, y')
title('Transient response')
grid on;

figure()
T=SYS;
sigma(T)
grid on;
title('Complementary sensitivity function');

figure()
S=eye(5)-SYS;
sigma(S)
grid on;
title('Sensitivity function');


figure()
S=eye(5)-SYS;
sigma(inv(S)*T)
grid on;
title('Loop Gain SVs');



SYSC=ss(Ac,Bc,Cc,Dc);
w=logspace(-3,4,500);
figure()
sigma(SYSC,w)
grid on;
title('Controller frequency response');

figure()
sigma(SYSC*S);
grid on;
title('Control sensitivity function');


Bclw=[ones(15,1);zeros(5,1)];
Dclw=[zeros(5,1)];


figure();

SYSw=ss(Acl,Bclw,Ccl,Dclw)
step(SYSw,t);
grid on;
ylabel('y')
title('Disturbance rejection');

% control effort for tracking
Cclu=[-Ko Cc];
Dclu=0*eye(5);
SYSu=ss(Acl,Bcl,Cclu,Dclu);

figure()
step(SYSu,[t])
grid on;
ylabel('u')
title('Control effort for tracking');

%%%%% Control Effort for Disturbance Rejection
SYSuw=ss(Acl,Bclw,Cclu,Dclw)
figure()
step(SYSuw,t)
grid on;
xlabel('Time (msec)');
ylabel('u')
title('Control effort for disturbance rejection');


r=[30*(0:0.01:10) 300*ones(size(0.01:0.01:10))];
t=0:0.01:20;

figure()
rr=[r;r;r;r;r];
[yr,t]=lsim(SYS,rr,t)
plot(t,r)
hold on;
plot(t,yr)
xlabel('Time (sec)');
ylabel('r,y');
hold off
grid

r=[30*(0:0.01:10) 300*ones(size(0.01:0.01:10))];
t=0:0.01:20;



% Control Effort

figure()
rr=[r;r;r;r;r];
[yu,t]=lsim(SYSu,rr,t)
plot(t,yu)
xlabel('Time (sec)');
ylabel('u');
hold off
grid

return



%%%%%%%%%%%%%% Add Feedforward %%%%%%%%%%%%%%%
%%%%%%%
%%%%%%%%
N=eye(2);

%%% r to y
AclN=[A-B*Ko B*Cc;
    Bc*C Ac];
BclN=[B*N;-Bc];
CclN=[C zeros(2,2)];
DclN=[0*eye(2)];

%%% r to e
Ccle=[C zeros(2,2)];
Dcle=-eye(2);
%%%% transmission zeros
tzero(AclN,BclN,CclN,DclN)

t=0:.01:25;

SYSN=ss(AclN,BclN,CclN,DclN);

figure()
step(SYSN,t)
xlabel('Time (msec)');
ylabel('r, y')
title('Transient response with feedforward')
grid on;

figure()
TN=SYSN;
sigma(TN)
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
GM1=20*log10(GM1)

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
GM1=20*log10(GM2)