include("odd_parity.jl")
using .OddParity
using Plots
using DelimitedFiles

T = Float64
l = 2
om_M_vals = [1e-3, 1e-2, 1e-1, 1.0]
C_vals = range(T(0.05), T(0.245), length=30)

results = []
push!(results, ["Compactness", "omega_M", "omega", "M_tov", "h_ratio_Odd"])

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
            h_ratio = OddParity.eval_h_ratio(p0, l, omega)
            push!(results, [C, om_M, omega, M, h_ratio])
            push!(plot_dict[om_M], h_ratio)
        end
    catch e
        println("Skipping C = ", C, " due to error: ", e)
    end
end

open("odd_parity_hratio.csv", "w") do io
    writedlm(io, results, ',')
end
println("Saved dataset to odd_parity_hratio.csv")

p = plot(title="Odd Parity h'/h vs Compactness (l=2)", 
         xlabel="Compactness (M/R)", 
         ylabel="h_ratio", 
         legend=:topleft)

for om_M in om_M_vals
    plot!(p, C_plot, plot_dict[om_M], label="omega*M = $om_M", linewidth=2, marker=:circle, markersize=3)
end

savefig(p, "odd_parity_hratio_plot.png")
println("Saved plot to odd_parity_hratio_plot.png")
