// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.01;
s0 = news;
Rectangle(s0) = {0.0, 0.0, 0.0, 1.0, 1.0};
s1 = news;
Rectangle(s1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2};
s2 = news;
Physical Surface(2) = {s1};
Physical Surface(0) = {s0};
bo1[] = BooleanFragments{ Surface{s0}; Delete; } { Surface{s1};};
