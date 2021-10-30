// This code was created by pygmsh vunknown.
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
Physical Surface(2) = {s1};
Physical Surface(3) = {s2};
Physical Surface(4) = {s3};
Physical Surface(5) = {s4};
Physical Surface(0) = {s0};
bo1[] = BooleanFragments{ Surface{s0}; Delete; } { Surface{s1};Surface{s2};Surface{s3};Surface{s4}; Delete;};
vb1[] = Boundary{Surface{ s1 };};
                        vb2[] = Boundary{Surface{ s2 };};
                        vb3[] = Boundary{Surface{ s3 };};
                        vb4[] = Boundary{Surface{ s4 };};
                        vb0[] = Boundary{Surface{ s0 };};
Physical Curve(5) = {vb0[],
        vb1[0], vb1[2],
        vb2[0], vb2[2],
        vb3[0], vb3[2],
        vb4[0], vb4[2]};
Physical Curve(5) -= {-vb1[1], -vb2[1], -vb3[3], -vb4[3]};
                         Physical Curve(1) = {vb1[3]};
                         Physical Curve(2) = {vb3[1]};
                         Physical Curve(3) = {vb2[3]};
                         Physical Curve(4) = {vb4[1]};
