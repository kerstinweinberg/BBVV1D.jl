module BBVV1D

using Printf, WriteVTK, LinearAlgebra, SparseArrays

export PointCloud, MaterialParameters, VelocityBC, simulation, read_vtk
export BBMaterial, CCMaterial, CCFMaterial, BAQPMaterial, OSBMaterial, default_material
export NoCorrection, EnergySurfaceCorrection
export const_one_kernel, linear_kernel, cubic_b_spline_kernel
export BeamSystem, MittlereKruemmung, BindungsKruemmung, knotenwerte, kruemmung,
       formaenderungsarbeit, statik, eigenfrequenzen, stabiler_zeitschritt, balken_dynamik
export ElasticaSystem, elastica_lagen, elastica_energie, elastica_residuum,
       elastica_gleichgewicht, elastica_gerade, elastica_knickform, elastica_pfad,
       elastica_massen, elastica_omega_max, elastica_dynamik

include("simulation.jl")
include("balken.jl")
include("elastica.jl")

include("VtkReader.jl")
using .VtkReader

end
