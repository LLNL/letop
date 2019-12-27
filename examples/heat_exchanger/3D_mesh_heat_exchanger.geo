// This code was created by pygmsh v6.0.2.
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = 0.05;
Mesh.CharacteristicLengthMax = 0.05;
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
bo1[] = BooleanFragments{ Volume{vol0}; Delete; } { Volume{vol1};Volume{vol2};Volume{vol3};Volume{vol4}; Delete;};

vol1Boundaries[] = Boundary { Volume {vol1}; };
vol2Boundaries[] = Boundary { Volume {vol2}; };
vol3Boundaries[] = Boundary { Volume {vol3}; };
vol4Boundaries[] = Boundary { Volume {vol4}; };
vol0Boundaries[] = Boundary { Volume {vol0}; };
Physical Surface(1) = {vol1Boundaries[0]};
Physical Surface(3) = {vol2Boundaries[0]};
Physical Surface(2) = {vol3Boundaries[1]};
Physical Surface(4) = {vol4Boundaries[1]};
Physical Surface(5) = {vol0Boundaries[]};
Physical Volume(0) = {vol0, vol3, vol4};
Physical Volume(2) = {vol1};
Physical Volume(3) = {vol2};
