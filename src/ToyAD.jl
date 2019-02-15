module ToyAD

export Dual, dualseed, value, partials
export @add_forward_unary, @add_forward_binary
export @partials

include("types.jl")
include("rules.jl")
include("forward.jl")

end
