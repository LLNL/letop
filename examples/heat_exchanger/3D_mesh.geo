// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.2;
Mesh.CharacteristicLengthMax = 0.2;
v0 = news;
Box(v0) = {0.0, 0.0, 0.0, 1.2, 1.5, 0.8};
v1 = news;
Box(v1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2, 0.2};
v2 = news;
Box(v2) = {-0.2, 0.46, 0.0, 0.2, 0.2, 0.2};
v3 = news;
Box(v3) = {1.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2, 0.2};
v4 = news;
Box(v4) = {1.2, 0.46, 0.0, 0.2, 0.2, 0.2};


bo1[] = BooleanFragments{ Volume{v0}; Delete; } { Volume{v1};Volume{v2};Volume{v3};Volume{v4}; Delete;};//+

Physical Volume(2) = {v1};
Physical Volume(3) = {v2};
Physical Volume(0) = {v0, v4, v3};

vb1[] = Boundary{Volume{v1};};
vb2[] = Boundary{Volume{v2};};
vb3[] = Boundary{Volume{v3};};
vb4[] = Boundary{Volume{v4};};
vb0[] = Boundary{Volume{v0};};

For ii In { 0 : (#vb1[] -1) }
	Printf("boundary number %g = %g", ii, vb1[ii]);
EndFor
Physical Surface(5) = {vb0[], 
			vb1[2], vb1[3], vb1[4], vb1[5],
			vb2[2], vb2[3], vb2[4], vb2[5],
			vb3[2], vb3[3], vb3[4], vb3[5],
			vb4[2], vb4[3], vb4[4], vb4[5]
			};
Physical Surface(5) -= {vb1[1], vb2[1], vb3[0], vb4[0]};

//+
Physical Surface(1) = {vb1[0]};
//+
Physical Surface(2) = {vb3[1]};
//+
Physical Surface(3) = {vb2[0]};
//+
Physical Surface(4) = {vb4[1]};
//+
