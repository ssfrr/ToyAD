module PedagogicalAutoDiff

export Dual, dualseed, value, partials
export @partials

include("types.jl")
include("rules.jl")
include("forward.jl")

end
