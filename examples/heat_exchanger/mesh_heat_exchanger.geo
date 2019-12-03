// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.08;
Mesh.CharacteristicLengthMax = 0.08;
s0 = news;
Rectangle(s0) = {0.0, 0.0, 0.0, 1.2, 1.5};
s1 = news;
Rectangle(s1) = {-0.2, 0.009999999999999981, 0.0, 0.2, 0.2};
s2 = news;
Rectangle(s2) = {-0.2, 0.24999999999999997, 0.0, 0.2, 0.2};
s3 = news;
Rectangle(s3) = {1.2, 0.009999999999999981, 0.0, 0.2, 0.2};
s4 = news;
Rectangle(s4) = {1.2, 0.24999999999999997, 0.0, 0.2, 0.2};
Physical Surface(2) = {s1};
Physical Surface(3) = {s2};
Physical Surface(4) = {s3};
Physical Surface(5) = {s4};
Physical Surface(0) = {s0};
bo1[] = BooleanUnion{ Surface{s0}; Delete; } { Surface{s1};Surface{s2};Surface{s3};Surface{s4}; Delete;};
Physical Curve(1) = {8};
//+
Physical Curve(3) = {12};
//+
Physical Curve(2) = {14};
//+
Physical Curve(4) = {18};
//+
Physical Curve(5) = {21, 5, 28, 7, 9, 11, 26, 27, 25, 24, 19, 17, 23, 15, 13, 22};
