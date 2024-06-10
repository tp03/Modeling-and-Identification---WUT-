%inicjalizacja zmiennych
x = [-1:0.01:1];
a0 = 0.00447127;
a1 = 0.108205;
a2 = 0.683702;
b0 = 0.00201207;
alfa1 = 1.2;
alfa2 = 0.56;
alfa3 = -1.45;
alfa4 = 1.15;

%charakterystyka statyczna nieliniowa
p = [b0*alfa4/a0, b0*alfa3/a0, b0*alfa2/a0, b0*alfa1/a0, 0];
y = polyval(p, x);

%charakterystyka statyczna liniowa
u_lin = 0.5;
a = (b0/a0)*(alfa1 + 2*alfa2*u_lin + 3*alfa3*u_lin^2 + 4*alfa4*u_lin^3);
b = (b0/a0)*(-alfa2*u_lin^2-2*alfa3*u_lin^3-3*alfa4*u_lin^4);
q = [a, b];
y2 = polyval(q, x);

%wykresy stat.
set (0 , 'defaulttextinterpreter' , 'latex') ;
set (0 , 'DefaultLineLineWidth' ,1) ;
set (0 , 'DefaultStairLineWidth' ,1) ;

figure(1);
plot(x, y);
xlabel('u');
ylabel('y');
print('char_stat_nielin.png' , '-dpng'   , '-r400');

figure(2);
plot(x, y);
xlabel('u');
ylabel('y');
hold on;
plot(x, y2);
legend('charakterystyka statyczna nieliniowa','charakterystyka statyczna zlinearyzowana')
print('char_stat_lin3.png' , '-dpng'   , '-r400');

T = 2.5;

%wykresy zadanie 6
sim("modi1_6.slx")
figure(3);
hold on;
plot(out.linear.time, out.linear.signals.values);
hold on;
plot(out.nonlinear.time, out.nonlinear.signals.values);
hold on;
legend({'charakterystyka liniowa', 'charakterystyka nieliniowa'}, 'Location', 'southeast');
xlabel('t(s)');
ylabel('y');
print('char_dyn_ciagly9.png' , '-dpng'   , '-r400')


%wykresy zadanie 8
sim("modi1_8.slx")
figure(3);
hold on;
plot(out.continuous.time, out.continuous.Data);
hold on;
stairs(out.discrete.time, out.discrete.Data);
hold on;
legend({'charakterystyka ciągła', 'charakterystyka dyskretna'}, 'Location', 'southeast');
xlabel('t(s)');
ylabel('y');
print('char_dyskr1.png' , '-dpng'   , '-r400')

%zmienne symboliczne
syms s a0 a1 a2 b0 alfa1 alfa2 alfa3 alfa4 u_lin

%macierze modelu przestrzeni stanu
A = [-a2 1 0; -a1 0 1; -a0 0 0];
B = [0 0 b0*(alfa1 + 2*alfa2*u_lin + 3*alfa3*u_lin^2 + 4*alfa4*u_lin^3)].';
C = [1 0 0];
D = 0;

%obliczenie tranmitancji
inA = inv(s*eye(3) - A);
G = C*inA*B+D;

%Obliczenie wzmocnień statycznych
K1 = subs(G, {s, a0, a1, a2, b0, alfa1, alfa2, alfa3, alfa4, u_lin}, {0, 0.00447127, 0.108205, 0.683702, 0.00201207, 1.2, 0.56, -1.45, 1.15, -0.2});
K2 = subs(G, {s, a0, a1, a2, b0, alfa1, alfa2, alfa3, alfa4, u_lin}, {0, 0.00447127, 0.108205, 0.683702, 0.00201207, 1.2, 0.56, -1.45, 1.15, 0.5});
K3 = subs(G, {s, a0, a1, a2, b0, alfa1, alfa2, alfa3, alfa4, u_lin}, {0, 0.00447127, 0.108205, 0.683702, 0.00201207, 1.2, 0.56, -1.45, 1.15, 0.9});