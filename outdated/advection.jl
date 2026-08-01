"""
Active flux method
https://deepblue.lib.umich.edu/bitstream/handle/2027.42/90701/AIAA-2011-3840-631.pdf

∂ₜu + ∂ₓu = 0 with periodic boundary
"""

using KitBase, Plots
using KitBase.OffsetArrays
using KitBase.ProgressMeter: @showprogress

begin
    tend = 1.0
    c = 0.5 # CFL
    ηn = 4 - 3 * c # η_near
    ηf = 3 * c - 2 # η_far
    ϕn = 2 - c # ϕ_near
    ϕf = c - 1 # ϕ_far
end

ps = PSpace1D(0, 1, 100, 1)

uc = sin.(2π * ps.x)
uf = OffsetArray{Float64}(undef, 0:ps.nx+2)
for i = 0:ps.nx+1
    uf[i] = sin.(2π * (ps.x[i] - ps.dx[i] / 2))
end
uf[end] = sin.(2π * (ps.x[end] + ps.dx[end] / 2))

ufb = zeros(ps.nx + 1)
uc0 = deepcopy(uc)
uf0 = deepcopy(uf)

dt = c * ps.dx[1]
nt = tend / dt |> Int

@showprogress for iter = 1:nt
    # inrerface values
    for i = 1:ps.nx+1
        uf[i] = uf0[i] - c * ηf * (uc[i-1] - uf0[i-1]) - c * ηn * (uf0[i] - uc[i-1])
        ufb[i] = uf0[i] - c * ϕf * (uc[i-1] - uf0[i-1]) - c * ϕn * (uf0[i] - uc[i-1])
    end

    # cell values
    for i = 1:ps.nx
        uc[i] += c * (ufb[i] - ufb[i+1])
    end

    # auxiliary
    uc[0] = uc[ps.nx]
    uc[ps.nx+1] = uc[1]
    uf[0] = uf[ps.nx]
    uf[ps.nx+2] = uf[2]
    uf0 .= uf
end

plot(ps.x[1:ps.nx], uc[1:ps.nx])
plot!(ps.x[1:ps.nx], uc0[1:ps.nx])
