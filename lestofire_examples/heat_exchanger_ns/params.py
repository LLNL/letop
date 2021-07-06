height = 1.5
shift_center = 0.52
dist_center = 0.03
inlet_width = 0.2
inlet_depth = 0.2
width = 1.2
line_sep = height / 2.0 - shift_center
# Line separating both inlets
DOMAIN = 0
INMOUTH1 = 2
INMOUTH2 = 3
OUTMOUTH1 = 4
OUTMOUTH2 = 5
INLET1, OUTLET1, INLET2, OUTLET2, WALLS, DESIGNBC = 1, 2, 3, 4, 5, 6
ymin1 = line_sep - (dist_center + inlet_width)
ymin2 = line_sep + dist_center
