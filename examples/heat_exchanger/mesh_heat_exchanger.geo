// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.02;
Mesh.CharacteristicLengthMax = 0.02;
s0 = news;
Rectangle(s0) = {0.0, 0.0, 0.0, 1.2, 1.5};
s1 = news;
Rectangle(s1) = {-0.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2};
s2 = news;
Rectangle(s2) = {-0.2, 0.26, 0.0, 0.2, 0.2};
s3 = news;
Rectangle(s3) = {1.2, -2.7755575615628914e-17, 0.0, 0.2, 0.2};
s4 = news;
Rectangle(s4) = {1.2, 0.26, 0.0, 0.2, 0.2};
bo1[] = BooleanUnion{ Surface{s0}; Delete; } { Surface{s1};Surface{s2};Surface{s3};Surface{s4}; Delete;};//+
Physical Surface(2) = {s1};
Physical Surface(3) = {s2};
Physical Surface(4) = {s3};
Physical Surface(5) = {s4};
Physical Surface(0) = {s0, s4, s3};
Physical Curve(5) = {27, 9, 11, 26, 21, 23, 29, 30, 24, 25, 19, 17, 34, 32};
//+
Physical Curve(1) = {31};
//+
Physical Curve(2) = {33};
//+
Physical Curve(3) = {12};
//+
Physical Curve(4) = {18};
