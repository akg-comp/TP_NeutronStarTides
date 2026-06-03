include("qnm_finder.jl")
using .QNMFinder

function run_full_verification()
    l = 2
    target_C = 0.14
    println("--- QNM Finder Reliability Verification ---")
    p0 = OddParity.find_p0_for_compactness(target_C, 1e-5, 1e-4)
    
    # Fundamental mode root finding
    omega_0 = 0.05  - 0.01im
    omega_1 = 0.051 - 0.01im
    omega_2 = 0.049 - 0.011im
    
    f_root = (w) -> QNMFinder.qnm_root_func(w, p0, l)
    
    qnm_w, f_val, iters = QNMFinder.muller_method(f_root, omega_0, omega_1, omega_2)
    
    println("\nConverged to ω = ", qnm_w)
    
    # Run Reliability Suite
    QNMFinder.verify_reliability(qnm_w, p0, l)
    
    # Scan for next w-mode (w_II)
    # Trying a significantly higher imaginary part for w_II
    println("\n--- Scanning for w_II (Highly Damped) ---")
    guess_w2 = [0.2 + 0.3im, 0.21 + 0.3im, 0.2 + 0.31im]
    qnm_w2, f_val2, iters2 = QNMFinder.muller_method(f_root, guess_w2[1], guess_w2[2], guess_w2[3]; max_iters=100)
    
    if abs(qnm_w - qnm_w2) > 1e-3
        println("Found potential higher mode: ω_II = ", qnm_w2)
        QNMFinder.verify_reliability(qnm_w2, p0, l)
    else
        println("Could not find a distinct higher mode with current guess.")
    end
end

run_full_verification()
