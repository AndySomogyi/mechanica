cl__1 = 1;

Mesh.CharacteristicLengthMin = 3;
Mesh.CharacteristicLengthMax = 3;
    
Point(1) = {0, 0, 0, 1};
Point(2) = {0.5, 0, 0, 1};
Point(3) = {-0.5, 0, 0, 1};
Point(4) = {0, 0.5, 0, 1};
Point(5) = {0, -0.5, 0, 1};
Point(6) = {0, 0, 0.5, 1};
Point(7) = {0, 0, -0.5, 1};
Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 2};
Circle(5) = {2, 1, 6};
Circle(6) = {6, 1, 3};
Circle(7) = {3, 1, 7};
Circle(8) = {7, 1, 2};
Circle(9) = {4, 1, 6};
Circle(10) = {6, 1, 5};
Circle(11) = {5, 1, 7};
Circle(12) = {7, 1, 4};
Line Loop(14) = {2, 7, 12};
Surface(14) = {14};
//Recombine Surface {14};
Line Loop(16) = {2, -6, -9};
Surface(16) = {16};
//Recombine Surface {16}
Line Loop(18) = {3, -10, 6};
Surface(18) = {18};
//Recombine Surface {18}
Line Loop(20) = {3, 11, -7};
Surface(20) = {20};
//Recombine Surface {20}
Line Loop(22) = {4, -8, -11};
Surface(22) = {22};
//Recombine Surface {22}
Line Loop(24) = {4, 5, 10};
Surface(24) = {24};
//Recombine Surface {24}
Line Loop(26) = {1, 9, -5};
Surface(26) = {26};
//Recombine Surface {26}
Line Loop(28) = {1, -12, 8};
Surface(28) = {28};
//Recombine Surface {28}
//Surface Loop(30) = {14, 16, 18, 20, 22, 24, 26, 28};
//Volume(30) = {30};

//num[]=Extrude{Surface{28}; Layers{{1} {1}} ; Recombine;}
Extrude{Surface{14,-16, -18, 20, 22, -24, -26, 28};Layers{{1},{0.2}}; Recombine;}