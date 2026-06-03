include("odd_parity.jl")
include("qnm_finder.jl")

function convergence_study()
    l = 2
    target_C = 0.14
    p0 = OddParity.find_p0_for_compactness(target_C, 1e-5, 1e-4)
    
    # Found mode
    omega_guess = 0.14673521862277875 + 0.12167726672406429im
    
    println("--- QNM Convergence Study (max_steps) ---")
    println("Target Mode: ", omega_guess)
    
    steps_list = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for n in steps_list
        # Re-evaluate the root function at the converged point with different steps
        residual = QNMFinder.evaluate_CF(omega_guess, l, 0.0, 0.0; max_steps=n) # Note: evaluate_CF needs M, R
        # Wait, qnm_root_func handles the matching. Let's use it directly but modify evaluate_CF call inside or use a wrapper.
        
        # Let's just solve again with Muller but varying the internal steps.
        # I'll create a local version of the root function for this test.
        k, R, M = QNMFinder.eval_k(p0, l, omega_guess)
        cf = QNMFinder.evaluate_CF(omega_guess, l, M, R; max_steps=n)
        val = k + cf
        println("Steps: $n, |k + C(omega)| = $(abs(val))")
    end
end

function find_more_modes()
    l = 2
    target_C = 0.14
    p0 = OddParity.find_p0_for_compactness(target_C, 1e-5, 1e-4)
    f_root = (w) -> QNMFinder.qnm_root_func(w, p0, l)

    # Guess for w1 (higher frequency, higher damping)
    # Usually real and imaginary parts both increase.
    # Try Re ~ 0.25, Im ~ 0.2
    guesses = [
        (0.25 - 0.2im, 0.251 - 0.2im, 0.249 - 0.201im),
        (0.4 - 0.3im, 0.401 - 0.3im, 0.399 - 0.301im)
    ]
    
    println("\n--- Searching for Higher Modes ---")
    for (i, g) in enumerate(guesses)
        println("\nSearching for Mode w_$i...")
        qnm_w, f_val, iters = QNMFinder.muller_method(f_root, g[1], g[2], g[3]; max_iters=50)
        println("Result w_$i: ω = $qnm_w, residual = $(abs(f_val))")
    end
end

convergence_study()
find_more_modes()
