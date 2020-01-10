// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.02;
Mesh.CharacteristicLengthMax = 0.02;
s0 = news;
Box(s0) = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
s1 = news;
Box(s1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2, 0.2};
s2 = news;
bo1[] = BooleanFragments{ Volume{s0}; Delete; } { Volume{s1}; Delete;};
Physical Volume(2) = {s1};
Physical Volume(0) = {s0};
