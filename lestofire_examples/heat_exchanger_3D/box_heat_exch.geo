//+
SetFactory("OpenCASCADE");
//+
Box(1) = {0, 0, 0, 10, 10, 3};
//+
Cylinder(2) = {0, 2, 1.5, -2, 0, 0, 1.0, 2*Pi};
//+
Cylinder(3) = {0, 8, 1.5, -2, 0, 0, 1.0, 2*Pi};
//+
Cylinder(4) = {10, 8, 1.5, 2, 0, 0, 1.0, 2*Pi};
//+
Cylinder(5) = {10, 2, 1.5, 2, 0, 0, 1.0, 2*Pi};
//+
BooleanFragments{ Volume{1}; Delete; }{ Volume{4}; Volume{5}; Volume{2}; Volume{3}; Delete; }
Dilate {{0, 0, 0}, {0.1, 0.1, 0.1}} {
  Volume{3}; Volume{1}; Volume{2}; Volume{4}; Volume{5};
}
//+
Physical Surface(1) = {26};
//+
Physical Surface(2) = {40};
//+
Physical Surface(3) = {38};
//+
Physical Surface(4) = {42};
//+
Physical Surface(5) = {34, 35, 36, 28, 33, 30, 39, 41, 37, 25};
//+
Physical Volume(0) = {1};
mesh_size = 0.015;
MeshSize{ PointsOf{ Volume{1, 2, 3, 4, 5}; } } = mesh_size;
//+
Physical Volume(6) = {3};
//+
Physical Volume(7) = {2};
//+
Physical Volume(8) = {4};
//+
Physical Volume(9) = {5};
