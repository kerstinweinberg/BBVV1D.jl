module BBVV1D

using Printf, WriteVTK

export PointCloud, BondBasedMaterial, VelocityBC, simulation, read_vtk

include("simulation.jl")

include("VtkReader.jl")
using .VtkReader

end
