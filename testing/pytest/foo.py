r1 = Resistor(R=10)
c = Capacitor(C=0.01)
r2 = Resistor(R=100) 
l = Inductor(L=0.1)
ac = VsourceAC(V=220)
g = Ground()

bind (AC.p, R1.p); 
bind (R1.n, C.p); 
bind (C.n, AC.n); 
bind (R1.p, R2.p); 
bind (R2.n, L.p); 
bind (L.n, C.n); 
bind (AC.n, G.p);
