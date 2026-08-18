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


Q=C'*C;
R=eye(nu);

[K,S,ECLP]=lqr(A,B,Q,R)