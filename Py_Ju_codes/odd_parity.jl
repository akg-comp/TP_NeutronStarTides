module OddParity
    using DifferentialEquations

    T = Float64
    gc2 = T(7.42365e-29)
    G = T(6.6732e-8)
    cc = T(2.9979e10)
    Msun = T(1.47664)
    μ0 = T(1e-6)
    μsurf = T(1e-12)

    # --------------------------------------------------------------------------
    # Equation of State
    # --------------------------------------------------------------------------
    function ρ_eos(p1::T) where {T}
        K = T(100) * Msun^2
        γ0 = T(2)
        ρb = (p1 / K)^(T(1)/γ0)
        return ρb + p1
    end

    # --------------------------------------------------------------------------
    # Background (TOV) Solution exactly mirroring even-parity framework
    # --------------------------------------------------------------------------
    function tov_derivatives!(du, u, p_params, x)
        r_var, m_var, νb_var = u
        p_val = exp(x)
        ρ_val = ρ_eos(p_val)
        dmdr = 4 * T(pi) * r_var^2 * ρ_val
        dνdr = 2 * (m_var + 4 * T(pi) * r_var^3 * p_val) / (r_var * (r_var - 2 * m_var))
        dpdr = -(ρ_val + p_val) / 2 * dνdr
        drdx = p_val / dpdr
        du[1] = drdx; du[2] = drdx * dmdr; du[3] = drdx * dνdr
    end

    function solve_tov(p0_val::T)
        h1 = μ0 * p0_val
        x0 = log(p0_val - h1)
        ρ0 = ρ_eos(p0_val)
        r0 = sqrt(2 * h1 / (4 * T(pi) * (ρ0 + p0_val) * (ρ0/3 + p0_val)))
        m0 = (4 * T(pi) / 3) * ρ0 * r0^3
        νb0 = T(0)
        xf = log(μsurf * p0_val)
        prob = ODEProblem(tov_derivatives!, [r0, m0, νb0], (x0, xf))
        tol = 1e-10
        sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)
        r_f, m_f, νb_f = sol.u[end]
        ν0 = -νb_f + log(1 - 2 * m_f / r_f)
        return (; sol, x0, xf, M_tov=m_f, R_tov=r_f, ν0)
    end

    function target_compactness_func(p0_val::T, target_C::T)
        tov = solve_tov(p0_val)
        return (tov.M_tov / tov.R_tov) - target_C
    end

    function find_p0_for_compactness(target_C::T, p0_guess1::T, p0_guess2::T; tol::T=T(1e-8), max_iter::Int=50)
        p0_0 = p0_guess1
        p0_1 = p0_guess2
        f0 = target_compactness_func(p0_0, target_C)
        f1 = target_compactness_func(p0_1, target_C)
        for i in 1:max_iter
            if abs(f1) < tol return p0_1 end
            p0_new = p0_1 - f1 * (p0_1 - p0_0) / (f1 - f0)
            
            # Prevent unphysical negative pressure from secant method mathematically overshooting!
            if p0_new <= 0.0
                p0_new = p0_1 / 2.0
            end
            
            p0_0 = p0_1; f0 = f1; p0_1 = p0_new
            f1 = target_compactness_func(p0_1, target_C)
        end
        error("Secant method failed to converge for target compactness $target_C. Target likely exceeds the EOS theoretical maximum compactness.")
    end

    # --------------------------------------------------------------------------
    # Odd-Parity Perturbation
    # --------------------------------------------------------------------------
    function odd_derivatives!(du, u, p_params, x)
        tov_sol, l, omega, ν0_offset = p_params
        h, hp = u
        
        # Evaluate background geometry
        r, m, νb = tov_sol(x)
        ν = νb + ν0_offset
        p0 = exp(x)
        eps = ρ_eos(p0)
        
        # Geometry shortcuts
        exp_lam = 1.0 / (1.0 - 2.0 * m / r)
        exp_nu = exp(ν)
        exp_2nu = exp_nu * exp_nu
        exp_lam_nu = exp_lam * exp_nu
        exp_lam_minus_nu = exp_lam / exp_nu
        
        # Powers (chained to avoid allocation/power overhead)
        r2 = r * r
        om2 = omega * omega
        om2_r2 = om2 * r2
        om4_r4 = om2_r2 * om2_r2
        
        # Background derivatives
        nu_p = 2.0 * (m + 4.0 * T(pi) * r * r2 * p0) / (r * (r - 2.0 * m))
        p_prime = -(eps + p0) / 2.0 * nu_p
        drdx = p0 / p_prime
        
        # Exact paper components for Odd-Parity eq (53)
        Λl = T(l^2 + l - 2)
        D = Λl * exp_nu - om2_r2
        
        A_num = 4.0 * T(pi) * Λl * r * exp_lam_nu * (eps + p0) - 
                r * om2 * (exp_lam * (4.0 * T(pi) * r2 * (eps - p0) - 1.0) + 3.0)
                
        B_num = (1.0 / r2) * (
                    2.0 * (Λl * exp_nu + 2.0 * om2_r2) + 
                    exp_lam_minus_nu * (
                        Λl * exp_2nu * (Λl - 8.0 * T(pi) * r2 * (eps + p0)) - 
                        2.0 * om2_r2 * exp_nu * (Λl + 1.0 - 4.0 * T(pi) * r2 * (eps - p0)) + 
                        om4_r4
                    )
                )

        # ----------------------------------------------------------------------
        # OUT-OF-EQUILIBRIUM (OoE) PLACEHOLDERS
        # ----------------------------------------------------------------------
        # When fully tracking OoE effects, h(r) couples with h1(r) and U(r).
        # We set these to zero for now. Note: once activated, the ODE integration 
        # MUST migrate to complex variables since S_Source involves 'im * omega'.
        S_1A  = 0.0
        S_1Ap = 0.0
        S_Z   = 0.0
        
        # Evaluated Source terms spanning Eq 53 from the provided analytical draft:
        # S_source = (16.0*T(pi)*exp_nu / (im*omega*r)) * (Λl*exp_nu + exp_lam*(Λl*exp_nu*(1.0 - 4.0*T(pi)*r2*(eps - p0)) - 2.0*om2_r2*(1.0 - 2.0*T(pi)*r2*(eps - 3.0*p0))) + 2.0*om2_r2) * S_1A + 
        #            (16.0*T(pi)*exp_nu / (im*omega)) * D * S_1Ap - 
        #            (8.0*T(pi)*exp_lam / (im*omega*r2)) * D * S_Z
        S_source = 0.0
                
        h_double_prime = (A_num / D) * hp + (B_num / D) * h + (S_source / D)
        
        # Integrate with respect to x
        du[1] = drdx * hp
        du[2] = drdx * h_double_prime
    end

    function solve_odd(p0_val::T, l::Int, omega::Number)
        tov = solve_tov(p0_val)
        x0, xf, sol = tov.x0, tov.xf, tov.sol
        r0 = sol(x0)[1]
        
        # ----------------------------------------------------------------------
        # High Accuracy Frobenius Expansion Initial Conditions at r -> 0
        # ----------------------------------------------------------------------
        # The regular solution follows: h(r) = r^(l+1) + c2 * r^(l+3) + ...
        # Evaluating components accurately up to O(r^2):
        
        pc   = p0_val
        eps0 = ρ_eos(pc)
        exp_nuc = exp(tov.ν0)
        
        # Metric second derivatives terms: e^λ ≈ 1 + λ2*r^2; e^ν ≈ e^νc(1 + ν2*r^2)
        λ2 = 8.0 * T(pi) / 3.0 * eps0
        ν2 = 4.0 * T(pi) / 3.0 * (eps0 + 3.0 * pc)
        Λl = T(l^2 + l - 2)
        
        # Extracted limiting coefficients exactly derived from ODE limit balance:
        D0 = Λl * exp_nuc
        D2 = Λl * exp_nuc * ν2 - omega^2
        A1 = 4.0 * T(pi) * Λl * exp_nuc * (eps0 + pc) - 2.0 * omega^2
        B2 = exp_nuc * Λl * (ν2 * (2.0 + Λl) + Λl * λ2 - 8.0 * T(pi) * (eps0 + pc)) + 2.0 * omega^2 * (1.0 - Λl)
        
        # Solve for exact continuous c2 error term
        c2 = (A1 * (l + 1.0) + B2 - D2 * l * (l + 1.0)) / (2.0 * (2.0 * l + 3.0) * D0)
        
        # Set normalized Initial Value (H0=1)
        h0  = r0^(l + 1) * (1.0 + c2 * r0^2)
        hp0 = (l + 1) * r0^l + c2 * (l + 3) * r0^(l + 2)
        
        prob = ODEProblem(odd_derivatives!, [h0, hp0], (x0, xf), (sol, l, omega, tov.ν0))
        odd_sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-10)
        
        return (; sol=odd_sol, tov)
    end

    function eval_h_ratio(p0_val::T, l::Int, omega::Number)
        res = solve_odd(p0_val, l, omega)
        h_R, hp_R = res.sol.u[end]
        return hp_R / h_R
    end

    function eval_zeta(p0_val::T, l::Int, omega::Number)
        res = solve_odd(p0_val, l, omega)
        h_R, hp_R = res.sol.u[end]
        R = res.tov.R_tov
        M = res.tov.M_tov
        
        # Evaluate constants at R
        e_nu = 1.0 - 2.0 * M / R
        e_lam = 1.0 / e_nu
        Λl = T(l^2 + l - 2)
        
        om2 = omega^2
        R2 = R^2
        
        # RW Denominator mapping and its derivative
        D_R = Λl * e_nu - om2 * R2
        D_prime_R = 2.0 * M * Λl / R2 - 2.0 * om2 * R
        
        # Second derivative h''(R) using exterior vacuum Eq 53 exactly (ε=p=0 at boundary)
        A_surf = (R * om2 * (e_lam - 3.0)) / D_R
        B_surf = (2.0 * (Λl * e_nu + 2.0 * om2 * R2) + 
                  e_lam^2 * (Λl^2 * e_nu^2 - 2.0 * om2 * R2 * e_nu * (Λl + 1.0) + om2^2 * R2^2)) / (R2 * D_R)
        
        hpp_R = A_surf * hp_R + B_surf * h_R
        
        # Zeta Factor: Z_RW = f(r) * g(r) => (Z'/Z) = f'/f + g'/g
        # The complex amplitude matches cancel completely from the ratio!
        nu_prime_R = 2.0 * M / (R^2 * e_nu)
        f_ratio = 1.0 / R + nu_prime_R - D_prime_R / D_R
        
        g_R = hp_R - 2.0 * h_R / R
        g_prime_R = hpp_R - 2.0 * hp_R / R + 2.0 * h_R / R2
        g_ratio = g_prime_R / g_R
        
        zeta = 2.0 * M * (f_ratio + g_ratio)
        
        return zeta
    end

    function eval_response_BA(p0_val::T, l::Int, omega::Number)
        res = solve_odd(p0_val, l, omega)
        h_R, hp_R = res.sol.u[end]
        R = res.tov.R_tov
        M = res.tov.M_tov
        
        # Evaluate constants at R
        e_nu = 1.0 - 2.0 * M / R
        e_lam = 1.0 / e_nu
        Λl = T(l^2 + l - 2)
        
        om2 = omega^2
        R2 = R^2
        
        # RW Denominator mapping and its derivative
        D_R = Λl * e_nu - om2 * R2
        D_prime_R = 2.0 * M * Λl / R2 - 2.0 * om2 * R
        
        # Second derivative h''(R) using exterior vacuum Eq 53 exactly (ε=p=0 at boundary)
        A_surf = (R * om2 * (e_lam - 3.0)) / D_R
        B_surf = (2.0 * (Λl * e_nu + 2.0 * om2 * R2) + 
                  e_lam^2 * (Λl^2 * e_nu^2 - 2.0 * om2 * R2 * e_nu * (Λl + 1.0) + om2^2 * R2^2)) / (R2 * D_R)
        
        hpp_R = A_surf * hp_R + B_surf * h_R
        
        # Zeta Factor: Z_RW = f(r) * g(r) => (Z'/Z) = f'/f + g'/g
        nu_prime_R = 2.0 * M / (R^2 * e_nu)
        f_ratio = 1.0 / R + nu_prime_R - D_prime_R / D_R
        
        g_R = hp_R - 2.0 * h_R / R
        g_prime_R = hpp_R - 2.0 * hp_R / R + 2.0 * h_R / R2
        g_ratio = g_prime_R / g_R
        
        Z_ratio = f_ratio + g_ratio
        
        # B/A response
        BA = R^5 * (3.0 - R * Z_ratio) / (R * Z_ratio + 2.0)
        return BA
    end
end

using .OddParity

T = Float64
l = 2
omega = T(0.004)
target_C = T(0.14)

println("--- Starting Odd Parity Verification ---")
println("Finding p0 for Target Compactness C = ", target_C)
p0_c14 = OddParity.find_p0_for_compactness(target_C, T(1e-5), T(1e-4))
println("p0_c14 = ", p0_c14)

println("\nIntegrating Odd Parity ODE to the surface (l=", l, ", omega=", omega, ")...")
try
    res = OddParity.solve_odd(p0_c14, l, omega)
    u_surf = res.sol.u[end]
    println("Integration Successful!")
    println("Surface values at R = ", res.tov.R_tov, " :")
    println("  h(R)  = ", u_surf[1])
    println("  h'(R) = ", u_surf[2])
    
    zeta_odd = OddParity.eval_zeta(p0_c14, l, omega)
    println("Odd Parity Zeta = ", zeta_odd)
catch e
    println("Error evaluating odd parity integration! ", e)
    rethrow(e)
end
