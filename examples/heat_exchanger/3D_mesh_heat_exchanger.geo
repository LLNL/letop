// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.04;
Mesh.CharacteristicLengthMax = 0.04;
vol0 = newv;
Box(vol0) = {0.0, 0.0, 0.0, 1.2, 1.5, 1.5};
vol1 = newv;
Box(vol1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2, 0.2};
vol2 = newv;
Box(vol2) = {-0.2, 0.26, 0.0, 0.2, 0.2, 0.2};
vol3 = newv;
Box(vol3) = {1.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2, 0.2};
vol4 = newv;
Box(vol4) = {1.2, 0.26, 0.0, 0.2, 0.2, 0.2};
Physical Volume(2) = {vol1};
Physical Volume(3) = {vol2};
Physical Volume(0) = {vol0};
//Physical Surface(5) = {1, 2, 3, 4, 5, 12, 6, 18, 8, 15, 9, 30, 10, 14, 21, 25, 11, 16, 17, 22, 23, 24, 29, 28, 27};
//+
Physical Surface(1) = {7};
//+
Physical Surface(2) = {20};
//+
Physical Surface(3) = {13};
//+
Physical Surface(4) = {26};
