include("qnm_finder.jl")
using .QNMFinder
using .OddParity
using Plots

"""
    run_reliability_check(p0, l, omega_start, R_factors)

Computes the QNM frequency at different matching radii and plots the stability.
"""
function run_reliability_check(p0, l, omega_start, R_factors=[1.0, 1.1, 1.2, 1.3])
    roots = []
    residuals = []
    
    println("--- Reliability Check Starting ---")
    println("Targeting mode near: ", omega_start)
    
    # Track the current root to use as the next guess (trajectory tracking)
    current_guess = omega_start
    
    for Rf in R_factors
        println("\nEvaluating at R2 = $(Rf)*R_star...")
        
        # We use Muller's method to find the exact root at this R
        # We provide a tight neighborhood around the start guess
        w0 = current_guess * 0.99
        w1 = current_guess
        w2 = current_guess * 1.01
        
        f_target = (w) -> QNMFinder.qnm_root_func(w, p0, l; R_factor=Rf)
        
        try
            qnm, f_val, iters = QNMFinder.muller_method(f_target, w0, w1, w2; tol=1e-11)
            push!(roots, qnm)
            push!(residuals, abs(f_val))
            println("  Found Root: ", qnm)
            println("  Residual:   ", abs(f_val))
            
            # Update guess for next Rf to improve convergence
            current_guess = qnm
        catch e
            println("  Search failed at Rf=$Rf: ", e)
            push!(roots, NaN + NaN*im)
            push!(residuals, NaN)
        end
    end
    
    # 1. Plot Frequency Shifts relative to Rf=1.0
    ref_root = roots[1]
    deltas = [abs(r - ref_root) for r in roots]
    
    p1 = plot(R_factors, deltas, marker=:circle, 
              ylabel="|ω(R2) - ω(R_star)|", xlabel="R2 / R_star",
              title="QNM Matching Stability", label="Mode Shift",
              yscale=:log10, grid=:both)
              
    # 2. Plot Residuals
    p2 = plot(R_factors, residuals, marker=:square, color=:red,
              ylabel="|F(ω)| Residual", xlabel="R2 / R_star",
              title="Root Finding Precision", label="Residual",
              yscale=:log10, grid=:both)
              
    plot(p1, p2, layout=(2,1), size=(700, 800))
    savefig("qnm_reliability_plot.png")
    println("\nVerification complete. Results saved to 'qnm_reliability_plot.png'")
    
    return roots, R_factors
end

# Example usage for your 'Mode 1' tracking
function main()
    l = 2
    # From your Odd_new.ipynb: p0_c14 = 6.4643698065927e-5
    p0 = 6.4643698065927e-5 
    
    # Guessed from your trajectory plot (ωM ≈ 0.28 + 0.155im)
    # If C = 0.14, then R ≈ 7.14M, so ωR ≈ 2.0 + 1.1im
    # Adjust this guess to match your specific mode value
    omega_guess = 0.14673521862277875 + 0.12167726672406429im # Example baseline
    
    run_reliability_check(p0, l, omega_guess)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
