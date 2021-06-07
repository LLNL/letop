from .navier_stokes_brinkman import (
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    mark_no_flow_regions,
    InteriorBC,
)
from .advection_diffusion import AdvectionSUPG
from .utils import hs, min_mesh_size, calculate_max_vel, max_mesh_dimension
