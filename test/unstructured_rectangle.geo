//+
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {0, 0, 0, 1, 1.0, 0};
MeshSize{ PointsOf{ Surface{1}; } } = 0.02;
