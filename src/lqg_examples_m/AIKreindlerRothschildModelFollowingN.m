%%%%%%%%%%% Kreindler  and    Rothschild 1976 %%%%%%
%%% Model-following Example
%%%
%%% F-4 Aircraft
%%%
% Plant
Aa=[-1.768 0.4125 -14.25 0;
    -0.007 -0.3831 6.038 0;
    0.0016  -0.9975 -0.1551 0.0586;
    1 0 0 0];
eig(Aa)
Ba=[1.744 8.952;
   -2.92 -0.3075;
   0.0243 -0.0036;
   0 0];
C=[eye(4) zeros(4,6);
    zeros(4,6) eye(4)];
% Actuator Dynamics
Adelta=[ -20 0;
          0 -10];
Bdelta=[20 0;
        0 10];
Cdelta=eye(2);
% Model
Aaprime=[-4 0.865 -10 0;
    0.04 -0.507 5.87 0;
    0 -1 -0.743 0.0586;
    1 0 0 0];
Baprime=[3.3 20;
    -3.13 0;
    0 0;
    0 0];
eig(Aaprime)
% Command generator
tp=0.5;
wa=pi/(4*tp);
wr=wa;
sp=0.2;
k=exp(pi/4)*sp/(sin(pi/4));
delta=k*wr;

Ac=[0 1 0 0;
    -2*wr^2 -2*wr 0 0;
    0 0 0 1;
    0 0 -2*wa^2 -2*wa];
Cc=[1 0 0 0;
    0 0 1 0];

%%%% Implicit Model-following
alpha=1;
q1=0.1;
q2=10;
q3=0.1;
Qi=diag([q1 q2 q3 0 0 0 0 0]);

% Augmented system
A=[Aa Ba*Cdelta zeros(4,4);
    zeros(2,4) Adelta zeros(2,4);
    zeros(4,4) zeros(4,2) Ac];
B=[zeros(4,2);
    Bdelta;
    zeros(4,2)];

Am=[Aaprime Baprime*Cc;
    zeros(4,4) Ac];
Qhat=(C*A-Am*C)'*Qi*(C*A-Am*C);
N=B'*C'*Qi*(C*A-Am*C);
Nhat=N';
R=eye(2);
Rhat=R+B'*C'*Qi*C*B;

[K,S,CLP]=lqr(A,B,Qhat,Rhat,Nhat)
Kimp=K

t=0:.01:4;
U=zeros(2,401);
SYS=ss(A-B*K,B,C,zeros(8,2));

% Aileron command
X0=delta*[0 0 0 0 0 0 0 0 0 1]';
[Y,t,X]=lsim(SYS,U,t,X0);
figure()
plot(t,Y(:,1:4),'linewidth',2)
xlabel('Time (sec)');
grid on
legend('p',' r',' \beta',' \phi');
ylabel('p,r,\beta,\phi');
title('Implicit Model-following');
figure()
plot(t,X,'linewidth',2)
xlabel('Time (sec)');
legend
grid on


%%%%%%% Explicit Model-Following %%%%%%%

Am=Aaprime;
Bm=Baprime;

A=[Aa Ba*Cdelta zeros(4,8);
    zeros(2,4) Adelta zeros(2,8);
    zeros(4,6) Am Bm*Cc;
    zeros(4,6) zeros(4,4) Ac];

%%% Must put in 1e-3 in front of Bm, Bc, to get around numerical issues
%
B=[zeros(4,2);
    Bdelta;
    0.001*ones(4,2);
    0.001*ones(4,2)];

C=eye(4);

R=eye(2);
alpha=1;
Qe=alpha*eye(4);
Q=[C'*Qe*C zeros(4,2) -C'*Qe zeros(4,4);
    zeros(2,4) zeros(2,2) zeros(2,4) zeros(2,4);
    -Qe*C zeros(4,2) Qe zeros(4,4);
    zeros(4,4) zeros(4,2) zeros(4,4) zeros(4,4)];

[K,S,CLP]=lqr(A,B,Q,R)

Kexp=K

t=0:.01:4;
U=zeros(2,401);
CC=[eye(4) zeros(4,10)];
BB=[B(1:6,:);
    zeros(8,2)];
SYSE=ss(A-B*K,BB,CC,zeros(4,2));

% Aileron command
X0=delta*[0 0 0 0 0 0 0 0 0 0 0 0 0 1]';
[Y,t,X]=lsim(SYSE,U,t,X0);
figure()
plot(t,Y(:,1:4),'linewidth',2)
xlabel('Time (sec)');
grid on
legend('p',' r',' \beta',' \phi');
ylabel('p,r,\beta,\phi');
title('Explicit Model-following');

figure()
plot(t,X,'linewidth',2)
xlabel('Time (sec)');
legend
grid on


%%%%%% Pulse %%%%%
% Pilot Command
tp=0.5;
wa=pi/(4*tp);
wr=wa;
sp=0.2;
k=exp(pi/4)*sp/(sin(pi/4));
delta=k*wr;
Ac=[0 1;
    -2*wr^2 -2*wr];
Bc=[0;
    1];
Cc=[1 0];
sys=ss(Ac,Bc,Cc,0);
[y,t]=impulse(sys,t);
figure()
plot(t,y,'linewidth',2)
title('Pilot command')
grid on;
