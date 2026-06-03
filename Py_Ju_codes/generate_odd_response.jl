include("odd_parity.jl")
using .OddParity
using Plots
using DelimitedFiles

T = Float64
l = 2
om_M_vals = [1e-3, 1e-2]
C_vals = range(T(0.05), T(0.245), length=30)

results = []
push!(results, ["Compactness", "omega_M", "omega", "M_tov", "k2_mag"])

plot_dict = Dict(om => Float64[] for om in om_M_vals)
C_plot = Float64[]

for C in C_vals
    println("Processing C = ", C)
    
    p0 = 0.0
    M = 0.0
    try
        p0 = OddParity.find_p0_for_compactness(T(C), T(1e-6), T(1e-4))
        if p0 <= 0
            throw(DomainError(p0, "Negative p0 found"))
        end
        tov = OddParity.solve_tov(p0)
        M = tov.M_tov
        
        push!(C_plot, C)
        
        for om_M in om_M_vals
            omega = om_M / M
            BA = OddParity.eval_response_BA(p0, l, omega)
            R_surf = M / C
            k2_mag = 0.5 * (BA / (R_surf^5))
            push!(results, [C, om_M, omega, M, k2_mag])
            push!(plot_dict[om_M], k2_mag)
        end
    catch e
        println("Skipping C = ", C, " due to error: ", e)
    end
end

open("odd_parity_response.csv", "w") do io
    writedlm(io, results, ',')
end
println("Saved dataset to odd_parity_response.csv")

p = plot(title="Magnetic Love Number (k2_mag) vs Compactness", 
         xlabel="Compactness (C = M/R)", 
         ylabel="k2_mag", 
         legend=:topright)

C_ref = [0.05, 0.1, 0.15, 0.2, 0.25]
k_ref = [-0.002, -0.0012, -0.0007, -0.0003, -0.0001]
plot!(p, C_ref, k_ref, lw=2, ls=:dash, label="Damour-Nagar (ref)", color=:black)

for om_M in om_M_vals
    plot!(p, C_plot, plot_dict[om_M], label="omega*M = $om_M", linewidth=2, marker=:circle, markersize=3)
end

savefig(p, "odd_parity_response_plot.png")
println("Saved plot to odd_parity_response_plot.png")
