module BBVV1D

using Printf, WriteVTK

export PointCloud, MaterialParameters, VelocityBC, simulation, read_vtk
export BBMaterial, CCMaterial, CCFMaterial, BAQPMaterial, OSBMaterial, default_material
export NoCorrection, EnergySurfaceCorrection
export const_one_kernel, linear_kernel, cubic_b_spline_kernel

include("simulation.jl")

include("VtkReader.jl")
using .VtkReader

end
