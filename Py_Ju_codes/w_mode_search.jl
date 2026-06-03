include("odd_parity.jl")

module WModeSearch
    using LinearAlgebra
    using ..OddParity
    
    # ---------------------------------------------------------
    # Sotani Recurrence Coefficients (B8 - B11)
    # ---------------------------------------------------------
    function recurrence_coeffs(n, ω, l, M, a)
        α = (1.0 - 2.0*M/a) * n * (n + 1.0)
        β = -2.0 * (1im * ω * a + (1.0 - 3.0*M/a)*n) * n
        γ = (1.0 - 6.0*M/a) * n * (n - 1.0) + 6.0*M/a - l*(l + 1.0)
        δ = (2.0*M/a) * (n - 3.0) * (n + 1.0)
        return α, β, γ, δ
    end

    # ---------------------------------------------------------
    # 3-Term Reduction
    # ---------------------------------------------------------
    function build_tilde_coeffs(ω, l, M, a, a1_over_a0; N=200)
        α = zeros(ComplexF64, N+1)
        β = zeros(ComplexF64, N+1)
        γ = zeros(ComplexF64, N+1)
        δ = zeros(ComplexF64, N+1)
        
        for n in 0:N
            α[n+1], β[n+1], γ[n+1], δ[n+1] = recurrence_coeffs(n, ω, l, M, a)
        end
        
        α̂ = zeros(ComplexF64, N+1)
        β̂ = zeros(ComplexF64, N+1)
        γ̂ = zeros(ComplexF64, N+1)
        
        # Base B23
        α̂[1] = -1.0 + 0im
        β̂[1] = a1_over_a0
        γ̂[1] = 0.0 + 0im
        
        # Base B16
        α̂[2] = α[2]
        β̂[2] = β[2]
        γ̂[2] = γ[2]
        
        # Recurrence B17-19
        for n in 2:N
            γ̂_prev = γ̂[n]
            if abs(γ̂_prev) < 1e-30
                γ̂_prev = 1e-30 + 0im
            end
            factor = δ[n+1] / γ̂_prev
            
            α̂[n+1] = α[n+1]
            β̂[n+1] = β[n+1] - α̂[n] * factor
            γ̂[n+1] = γ[n+1] - β̂[n] * factor
        end
        
        return α̂, β̂, γ̂
    end

    # ---------------------------------------------------------
    # Backward CF Evaluator
    # ---------------------------------------------------------
    function evaluate_full_CF(ω, l, M, a, a1_over_a0; N=200)
        α̂, β̂, γ̂ = build_tilde_coeffs(ω, l, M, a, a1_over_a0; N=N)
        
        tiny = 1e-30 + 0im
        f = β̂[N+1]
        if abs(f) < 1e-30 f = tiny end
        
        for i in N:-1:1
            f = β̂[i] - (α̂[i] * γ̂[i+1]) / f
            if abs(f) < 1e-30 f = tiny end
        end
        
        return f
    end

    # ---------------------------------------------------------
    # Vacuum Matching Condition
    # ---------------------------------------------------------
    function a1_over_a0_from_interior(ψ, ψp, ω, a, M)
        f_val = 1.0 - 2.0*M/a
        # From Sotani (B14): a1/a0 = a * [ ψ'/ψ + iωa / (a - 2M) ]
        return a * (ψp/ψ + 1im*ω/f_val)
    end

    function qnm_root_func(ω, p0, l; N_CF = 200)
        # Integrate interior up to surface
        res = OddParity.solve_odd(p0, l, ω)
        ψR, ψpR = res.sol.u[end]
        R = res.tov.R_tov
        M = res.tov.M_tov
        
        a = R # As guaranteed by C=0.14 > 4M!
        a1_a0 = a1_over_a0_from_interior(ψR, ψpR, ω, a, M)
        
        f_root = evaluate_full_CF(ω, l, M, a, a1_a0; N=N_CF)
        return f_root
    end

    # ---------------------------------------------------------
    # Robust Muller Root Finder
    # ---------------------------------------------------------
    function muller_method(f::Function, x0::Number, x1::Number, x2::Number; max_iters=200, tol=1e-10)
        w = complex.([x0, x1, x2])
        fw = complex.([f(w[1]), f(w[2]), f(w[3])])
        
        x3 = w[3]
        fx3 = fw[3]
        
        println("  -> Initial guesses: ", w)
        for i in 1:max_iters
            h1 = w[2] - w[1]
            h2 = w[3] - w[2]
            
            delta1 = (fw[2] - fw[1]) / h1
            delta2 = (fw[3] - fw[2]) / h2
            
            a = (delta2 - delta1) / (h2 + h1)
            b = a * h2 + delta2
            c = fw[3]
            
            rad = sqrt(b^2 - 4.0 * a * c + 0im)
            den = abs(b + rad) > abs(b - rad) ? (b + rad) : (b - rad)
            if abs(den) < 1e-30 den = 1e-30 + 0im end
            
            dx = -2.0 * c / den
            x3 = w[3] + dx
            fx3 = f(x3)
            
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
end

using .WModeSearch

function main()
    l = 2
    target_C_start = 0.14
    target_C_end = 0.20
    C_steps = 5

    println("==================================================")
    println(" W-Mode QNM Search (C = $target_C_start to $target_C_end)")
    println("==================================================")

    # Initial guess for w-mode in M*omega format is generally higher freq than f-modes.
    # W-modes have high real frequencies and strong damping.
    # Let's seed an initial M*w guess. Re(M*w) ~ 0.3 - 0.5. Im(M*w) ~ 0.1 - 0.2
    
    # We sweep continuously down checking each Compactness
    C_range = range(target_C_start, target_C_end, length=C_steps)
    
    # Starting seed for the first Muller
    w_guess_base = 0.4 + 0.1im

    for C in C_range
        println("\nTarget Compactness: C = ", round(C, digits=4))
        p0 = WModeSearch.OddParity.find_p0_for_compactness(C, 1e-5, 1e-4)
        println("  Calculated p0 = ", p0)
        
        f_target = (ω) -> WModeSearch.qnm_root_func(ω, p0, l; N_CF = 250)
        
        # Perturb the guess slightly to generate 3 seed points
        w0 = w_guess_base
        w1 = w_guess_base * 1.01
        w2 = w_guess_base * 0.99 - 0.01im
        
        println("  Hunting for Root...")
        w_root, residual, iters = WModeSearch.muller_method(f_target, w0, w1, w2; tol=1e-10)
        
        println("  [FOUND] ω_QNM = ", w_root)
        println("  Residual = ", abs(residual), " | Iterations = ", iters)
        
        # Propagate the root forward as the guess for the next compactness iteration!
        w_guess_base = w_root
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
