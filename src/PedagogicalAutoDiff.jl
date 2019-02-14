module PedagogicalAutoDiff

export Dual, dualseed, value, partials

include("rules.jl")
include("forward.jl")

end
