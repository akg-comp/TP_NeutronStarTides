include("odd_parity.jl")

module QNMFinder
    using LinearAlgebra
    using ..OddParity
    import OrdinaryDiffEq: solve, ODEProblem

    function eval_k(p0_val::Float64, l::Int, omega::Number; R_factor=1.0)
        # 1. Solve interior to the surface R_star
        res = OddParity.solve_odd(p0_val, l, omega)
        h_S, hp_S = res.sol.u[end]
        R_star = res.tov.R_tov
        M = res.tov.M_tov

        # 2. Integrate vacuum ODE to R_match = R_star * R_factor
        R_match = R_star * R_factor
        
        h_match, hp_match = h_S, hp_S
        if R_factor > 1.0
            function vacuum_derivs!(du, u, p, r)
                h, hp = u
                M_v, l_v, omega_v = p
                e_nu = 1.0 - 2.0 * M_v / r
                e_lam = 1.0 / e_nu
                Λl = Float64(l_v^2 + l_v - 2)
                om2 = omega_v^2
                r2 = r^2
                D = Λl * e_nu - om2 * r2
                A = (r * om2 * (e_lam - 3.0)) / D
                B = (2.0 * (Λl * e_nu + 2.0 * om2 * r2) + 
                     e_lam^2 * (Λl^2 * e_nu^2 - 2.0 * om2 * r2 * e_nu * (Λl + 1.0) + om2^2 * r2^2)) / (r2 * D)
                du[1] = hp
                du[2] = A * hp + B * h
            end
            prob = ODEProblem(vacuum_derivs!, [h_S, hp_S], (R_star, R_match), (M, l, omega))
            sol = solve(prob, OddParity.Tsit5(), reltol=1e-12, abstol=1e-12)
            h_match, hp_match = sol.u[end]
        end

        # 3. Compute matching ratio a1/a0 at R_match
        e_nu = 1.0 - 2.0 * M / R_match
        e_lam = 1.0 / e_nu
        Λl = Float64(l^2 + l - 2)
        om2 = omega^2
        R2 = R_match^2
        
        D = Λl * e_nu - om2 * R2
        D_prime = 2.0 * M * Λl / R2 - 2.0 * om2 * R_match
        
        # h'' at R_match (vacuum)
        A_v = (R_match * om2 * (e_lam - 3.0)) / D
        B_v = (2.0 * (Λl * e_nu + 2.0 * om2 * R2) + 
               e_lam^2 * (Λl^2 * e_nu^2 - 2.0 * om2 * R2 * e_nu * (Λl + 1.0) + om2^2 * R2^2)) / (R2 * D)
        
        hpp_match = A_v * hp_match + B_v * h_match
        
        nu_prime = 2.0 * M / (R2 * e_nu)
        f_ratio = 1.0 / R_match + nu_prime - D_prime / D
        
        g_match = hp_match - 2.0 * h_match / R_match
        gp_match = hpp_match - 2.0 * hp_match / R_match + 2.0 * h_match / R2
        g_ratio = gp_match / g_match
        
        Z_ratio = f_ratio + g_ratio
        
        # Seed value a1/a0 = R_match * [ Z'/Z + iw*R_match/(R_match - 2M) ]
        chi_ratio = 1im * omega * R_match / (R_match - 2.0 * M)
        k = R_match * (Z_ratio + chi_ratio)
        
        return k, R_match, M
    end

    function evaluate_CF(omega::Number, l::Int, M::Float64, R_match::Float64; max_steps=5000, tol=1e-12)
        # Expansion point R_match
        c0 = 1.0 - 2.0*M/R_match
        c1 = 6.0*M/R_match - 2.0
        c2 = 1.0 - 6.0*M/R_match
        c3 = 2.0*M/R_match
        
        d0 = -2im*omega*R_match + 6.0*M/R_match - 2.0
        d1 = 2.0*(1.0 - 6.0*M/R_match)
        d2 = 6.0*M/R_match
        
        e0 = 6.0*M/R_match - l*(l+1)
        e1 = -6.0*M/R_match
        
        # Recurrence: alpha_n*a_{n+1} + beta_n*a_n + gamma_n*a_{n-1} + delta_n*a_{n-2} = 0
        alpha = (n::Int) -> n*(n+1)*c0
        beta  = (n::Int) -> (n-1)*n*c1 + n*d0
        gamma = (n::Int) -> (n-2)*(n-1)*c2 + (n-1)*d1 + e0
        delta = (n::Int) -> (n-3)*(n-2)*c3 + (n-2)*d2 + e1
        
        # Step 1: Reduce to 3-term recurrence using Gaussian elimination
        # n=1: hat_alpha[1]*a2 + hat_beta[1]*a1 + hat_gamma[1]*a0 = 0
        hat_alpha = zeros(ComplexF64, max_steps+1)
        hat_beta  = zeros(ComplexF64, max_steps+1)
        hat_gamma = zeros(ComplexF64, max_steps+1)
        
        hat_alpha[1] = alpha(1)
        hat_beta[1]  = beta(1)
        hat_gamma[1] = gamma(1)
        
        for n in 2:max_steps
            fac = delta(n) / hat_gamma[n-1]
            hat_alpha[n] = alpha(n)
            hat_beta[n]  = beta(n) - hat_alpha[n-1] * fac
            hat_gamma[n] = gamma(n) - hat_beta[n-1] * fac
        end
        
        # Step 2: Evaluate Continued Fraction for a1/a0 = -hat_gamma[1] / (hat_beta[1] - ...)
        # condition is k + C = 0 where C = hat_gamma[1] / (hat_beta[1] - hat_alpha[1]*hat_gamma[2] / ...)
        
        tiny = 1e-30
        
        # Fraction structure: b0 + a1/(b1 + a2/(b2 + ...))
        # Here: hat_beta[1] - hat_alpha[1]*hat_gamma[2] / (hat_beta[2] - ...)
        
        f_val = hat_beta[1]
        if abs(f_val) < tiny f_val = tiny + 0im end
        
        C = f_val
        D = 0.0 + 0im
        
        for n in 2:max_steps
            # a_n term in Lentz's: -hat_alpha[n-1] * hat_gamma[n]
            an_lentz = -hat_alpha[n-1] * hat_gamma[n]
            bn_lentz = hat_beta[n]
            
            D = bn_lentz + an_lentz * D
            if abs(D) < tiny D = tiny + 0im end
            
            C = bn_lentz + an_lentz / C
            if abs(C) < tiny C = tiny + 0im end
            
            D = 1.0 / D
            delta_val = C * D
            f_val = f_val * delta_val
            
            if abs(delta_val - 1.0) < tol break end
        end
        
        return hat_gamma[1] / f_val
    end

    function qnm_root_func(omega::Number, p0_val::Float64, l::Int; R_factor=1.0)
        # Consistent eval_k: Match and Exp Point unified at R_match
        k, R_match, M = eval_k(p0_val, l, omega; R_factor=R_factor)
        
        # Continued Fraction part also using R_match
        c_frac = evaluate_CF(omega, l, M, R_match)
        
        # Root condition: a1/a0 (Solve) + ContinuedFractionTail = 0
        return k + c_frac
    end

    function muller_method(f::Function, x0::Number, x1::Number, x2::Number; max_iters=100, tol=1e-10)
        w = [complex(x0), complex(x1), complex(x2)]
        fw = [f(w[1]), f(w[2]), f(w[3])]
        
        println("Muller Method starting...")
        
        x3 = w[3]
        fx3 = fw[3]
        
        for i in 1:max_iters
            h1 = w[2] - w[1]
            h2 = w[3] - w[2]
            
            delta1 = (fw[2] - fw[1]) / h1
            delta2 = (fw[3] - fw[2]) / h2
            
            a = (delta2 - delta1) / (h2 + h1)
            if isnan(a) || isinf(a) return w[3], fw[3], i end
            
            b = a * h2 + delta2
            c = fw[3]
            
            rad = sqrt(b^2 - 4.0 * a * c + 0im)
            
            if abs(b + rad) > abs(b - rad)
                den = b + rad
            else
                den = b - rad
            end
            
            if abs(den) < 1e-30 den = 1e-30 end
            
            dx = -2.0 * c / den
            x3 = w[3] + dx
            
            if abs(imag(x3)) < 1e-15
                x3 = real(x3) - 1e-12im
            end
            
            fx3 = f(x3)
            # println("Iter $i: omega = $x3, |f(omega)| = $(abs(fx3))")
            
            if abs(dx) < tol || abs(fx3) < tol
                return x3, fx3, i
            end
            
            w[1] = w[2]
            w[2] = w[3]
            w[3] = x3
            
            fw[1] = fw[2]
            fw[2] = fw[3]
            fw[3] = fx3
        end
        return x3, fx3, max_iters
    end

    function verify_reliability(omega::Number, p0_val::Float64, l::Int)
        println("\n--- RELIABILITY VERIFICATION ---")
        res_R1 = abs(qnm_root_func(omega, p0_val, l; R_factor=1.0))
        res_R11 = abs(qnm_root_func(omega, p0_val, l; R_factor=1.1))
        res_R12 = abs(qnm_root_func(omega, p0_val, l; R_factor=1.2))
        
        println("Residual at R_star:    ", res_R1)
        println("Residual at 1.1*R_star: ", res_R11)
        println("Residual at 1.2*R_star: ", res_R12)
        
        if res_R11 < 1e-6 && res_R12 < 1e-6
            println("VERDICT: QNM is PHYSICALLY VERIFIED (Matching point independent).")
        else
            println("VERDICT: QNM is potentially UNPHYSICAL or coordinate dependent.")
        end
        
        println("\nChecking CF Convergence (max_steps):")
        for steps in [100, 500, 2000]
            val = abs(evaluate_CF(omega, l, 0.0, 0.0; max_steps=steps)) # This will fail if M,R not passed correctly. 
            # Fix: eval_k gets M,R
            k, R, M = eval_k(p0_val, l, omega)
            cf = evaluate_CF(omega, l, M, R; max_steps=steps)
            println("  Steps $steps: $(abs(k+cf))")
        end
    end
end

using .QNMFinder

function main()
    l = 2
    target_C = 0.14
    println("--- QNM Finder: Odd Parity ---")
    println("Finding p0 for Target Compactness C = ", target_C)
    p0_c14 = OddParity.find_p0_for_compactness(target_C, 1e-5, 1e-4)
    println("p0_c14 = ", p0_c14)
    
    # Fundamental mode guess in geometric units M \approx 1.4 Msun
    # Real(M*omega) ~ 0.05 ... 0.1 => Re(omega) ~ 0.03 ... 0.08
    # Im(M*omega) ~ 0.01        => Im(omega) ~ 0.005
    
    omega_0 = 0.05  - 0.01im
    omega_1 = 0.051 - 0.01im
    omega_2 = 0.049 - 0.011im
    
    f_root = (w) -> QNMFinder.qnm_root_func(w, p0_c14, l)
    
    println("\nStarting Muller Root Search for QNM:")
    qnm_w, f_val, iters = QNMFinder.muller_method(f_root, omega_0, omega_1, omega_2)
    
    println("\n--- Final QNM Frequency ---")
    println("ω_QNM = ", qnm_w)
    println("F(ω) residual = ", f_val)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
