// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.1;
v0 = news;
Rectangle(v0) = {0.0, 0.0, 0.0, 1.2, 1.5};
v1 = news;
Rectangle(v1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2};
v2 = news;
Rectangle(v2) = {-0.2, 0.46, 0.0, 0.2, 0.2};
v3 = news;
Rectangle(v3) = {1.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2};
v4 = news;
Rectangle(v4) = {1.2, 0.46, 0.0, 0.2, 0.2};


bo1[] = BooleanFragments{ Surface{v0}; Delete; } { Surface{v1};Surface{v2};Surface{v3};Surface{v4}; Delete;};//+


vb1[] = Boundary{Surface{v1};};
vb2[] = Boundary{Surface{v2};};
vb3[] = Boundary{Surface{v3};};
vb4[] = Boundary{Surface{v4};};
vb0[] = Boundary{Surface{v0};};

For ii In { 0 : (#vb1[] -1) }
	Printf("boundary number %g = %g", ii, vb1[ii]);
EndFor
Physical Curve(1) = {vb1[3]};
Physical Curve(2) = {vb3[1]};
Physical Curve(3) = {vb2[3]};
Physical Curve(4) = {vb4[1]};


Physical Curve(5) = {vb0[], 
			vb1[2], vb1[0],
			vb2[2], vb2[0],
			vb3[2], vb3[0],
			vb4[2], vb4[0]
			};

Physical Curve(5) -= {-vb1[1], -vb2[1], -vb3[3], -vb4[3]};


Physical Surface(2) = {v1};
Physical Surface(3) = {v2};
Physical Surface(0) = {v0, v4, v3};
