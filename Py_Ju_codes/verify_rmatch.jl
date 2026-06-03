include("odd_parity.jl")
include("qnm_finder.jl")

function r_match_independence()
    l = 2
    target_C = 0.14
    p0 = OddParity.find_p0_for_compactness(target_C, 1e-5, 1e-4)
    
    # Use the found mode
    omega_guess = 0.14673521862277875 + 0.12167726672406429im
    
    println("--- QNM R_match Independence Test ---")
    
    # Normal matching at R_star
    k1, R1, M1 = QNMFinder.eval_k(p0, l, omega_guess)
    cf1 = QNMFinder.evaluate_CF(omega_guess, l, M1, R1)
    val1 = k1 + cf1
    
    println("At R_match = R_star ($R1): |f(omega)| = $(abs(val1))")
    
    # Matching at 1.1 R_star
    # To match at 1.1 R_star, we need to integrate the ODE further in vacuum.
    # In vacuum, h''(r) = A_surf * h' + B_surf * h.
    # We can use solve_odd and just let it integrate further? 
    # Current solve_odd integrates to xf = log(μsurf * p0). This is exactly the star surface.
    # Let's modify solve_odd or create a vacuum integrator.
    
    function integrate_to_r(p0_val, l, omega, r_target)
        res = OddParity.solve_odd(p0_val, l, omega)
        h_R, hp_R = res.sol.u[end]
        R_star = res.tov.R_tov
        M = res.tov.M_tov
        
        if r_target <= R_star
            return h_R, hp_R, R_star, M
        end
        
        # Integrate vacuum from R_star to r_target
        function vacuum_derivs!(du, u, p, r)
            h, hp = u
            M, l, omega = p
            e_nu = 1.0 - 2.0 * M / r
            e_lam = 1.0 / e_nu
            Λl = 2.0^2 + 2.0 - 2.0 # l=2
            om2 = omega^2
            r2 = r^2
            D = Λl * e_nu - om2 * r2
            A = (r * om2 * (e_lam - 3.0)) / D
            B = (2.0 * (Λl * e_nu + 2.0 * om2 * r2) + 
                 e_lam^2 * (Λl^2 * e_nu^2 - 2.0 * om2 * r2 * e_nu * (Λl + 1.0) + om2^2 * r2^2)) / (r2 * D)
            du[1] = hp
            du[2] = A * hp + B * h
        end
        
        prob = ODEProblem(vacuum_derivs!, [h_R, hp_R], (R_star, r_target), (M, l, omega))
        sol = DifferentialEquations.solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
        return sol.u[end][1], sol.u[end][2], r_target, M
    end

    h_v, hp_v, Rv, Mv = integrate_to_r(p0, l, omega_guess, 1.1 * R1)
    
    # Compute k at Rv
    om2 = omega_guess^2
    Rv2 = Rv^2
    e_nu_v = 1.0 - 2.0 * Mv / Rv
    e_lam_v = 1.0 / e_nu_v
    Λl = 2.0^2 + 2.0 - 2.0
    
    D_Rv = Λl * e_nu_v - om2 * Rv2
    D_prime_Rv = 2.0 * Mv * Λl / Rv2 - 2.0 * om2 * Rv
    
    A_v = (Rv * om2 * (e_lam_v - 3.0)) / D_Rv
    B_v = (2.0 * (Λl * e_nu_v + 2.0 * om2 * Rv2) + 
          e_lam_v^2 * (Λl^2 * e_nu_v^2 - 2.0 * om2 * Rv2 * e_nu_v * (Λl + 1.0) + om2^2 * Rv2^2)) / (Rv2 * D_Rv)
    
    hpp_v = A_v * hp_v + B_v * h_v
    nu_prime_v = 2.0 * Mv / (Rv2 * e_nu_v)
    f_ratio_v = 1.0 / Rv + nu_prime_v - D_prime_Rv / D_Rv
    g_ratio_v = (hpp_v - 2.0*hp_v/Rv + 2.0*h_v/Rv2) / (hp_v - 2.0*h_v/Rv)
    Z_ratio_v = f_ratio_v + g_ratio_v
    
    chi_ratio_v = -1im * omega_guess * Rv / (Rv - 2.0 * Mv)
    k_v = Rv * (Z_ratio_v - chi_ratio_v)
    
    cf_v = QNMFinder.evaluate_CF(omega_guess, l, Mv, Rv)
    val_v = k_v + cf_v
    
    println("At R_match = 1.1 R_star ($Rv): |f(omega)| = $(abs(val_v))")
end

r_match_independence()
