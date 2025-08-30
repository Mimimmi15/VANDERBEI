
using Pkg
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Printf")
Pkg.add("Distributions")
using CSV, DataFrames, LinearAlgebra, Statistics, Printf,Plots, Distributions


# =================================================================
# 1. Chargement des données
# =================================================================
 df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame)
#df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/GHCND_USW00023183_clean.csv", DataFrame)
d = df.day
T = df.avg_temp
#T = df.value
n = length(d)

# =================================================================
# 2. Construction de la matrice de conception
# =================================================================
X = hcat(
    ones(n),                           # x0
    d,                                 # x1 : linéaire
    cos.(2π .* d ./ 365.25),          # x2 : saison (cos)
    sin.(2π .* d ./ 365.25),          # x3 : saison (sin)
    cos.(2π .* d ./ (10.7*365.25)),   # x4 : cycle solaire (cos)
    sin.(2π .* d ./ (10.7*365.25))    # x5 : cycle solaire (sin)
)



# =================================================================
# 3. MÉTHODE : GRADIENT CONJUGUÉ
# =================================================================
function solve_conjugate_gradient(X, T; max_iter=1000, tol=1e-12)
    A = X' * X
    b = X' * T
    n_params = size(A, 1)
    
    x = zeros(n_params)
    r = b - A * x
    p = r
    rsold = r' * r
    
    for i in 1:max_iter
        Ap = A * p
        α = rsold / (p' * Ap)
        x = x + α * p
        r = r - α * Ap
        rsnew = r' * r
        
        if sqrt(rsnew) < tol
            break
        end
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    end
    return x
end

# =================================================================
# 4. RÉSOLUTION
# =================================================================
β_cg = solve_conjugate_gradient(X, T)
T_hat_cg = X * β_cg


# =================================================================
# 5. CALCUL DES MÉTRIQUES
# =================================================================
residuals_cg = T - T_hat_cg
amplitude_saison_cg = sqrt(β_cg[3]^2 + β_cg[4]^2)
amplitude_solaire_cg = sqrt(β_cg[5]^2 + β_cg[6]^2)

# =================================================================
# 6. AFFICHAGE DES RÉSULTATS
# =================================================================
println("="^75)
println("RÉSULTATS RÉGRESSION L2 - GRADIENT CONJUGUÉ")
println("="^75)

@printf("x0 (température moyenne base)  = %12.3f °F\n", β_cg[1])
@printf("x1 (pente réchauffement)       = %12.6f °F/j → %8.2f °F/siècle\n", 
        β_cg[2], β_cg[2] * 365.25 * 100)
@printf("x2 (saison cos)                = %12.3f\n", β_cg[3])
@printf("x3 (saison sin)                = %12.3f\n", β_cg[4])
@printf("x4 (solaire cos)               = %12.3f\n", β_cg[5])
@printf("x5 (solaire sin)               = %12.3f\n", β_cg[6])

println("\n" * "-"^55)
@printf("Amplitude saisonnière          = %12.3f °F\n", amplitude_saison_cg)
@printf("Amplitude cycle solaire        = %12.3f °F\n", amplitude_solaire_cg)


# =================================================================
# 8. COMPARAISON AVEC VANDERBEI
# =================================================================
println("\n" * "="^75)
println("COMPARAISON AVEC RÉFÉRENCE VANDERBEI (2012)")
println("="^75)
@printf("Vanderbei: Réchauffement = 2.8 °F/siècle\n")
@printf("Modèle_L2:            = %.2f °F/siècle\n", β_cg[2] * 365.25 * 100)
@printf("Écart relatif:           = %.1f%%\n", abs(β_cg[2] * 365.25 * 100 / 2.8 - 1) * 100)

@printf("\nVanderbei: Amplitude saisonnière = 21.5 °F\n")
@printf("Modèle_L2:            = %.1f °F\n", amplitude_saison_cg)
@printf("Écart relatif:           = %.1f%%\n", abs(amplitude_saison_cg / 21.5 - 1) * 100)

@printf("\nVanderbei: Amplitude solaire = 0.29 °F\n")
@printf("Modèle_L2:            = %.3f °F\n", amplitude_solaire_cg)
@printf("Écart relatif:           = %.1f%%\n", abs(amplitude_solaire_cg / 0.29 - 1) * 100)
println("="^75)


# =================================================================
# 9. FONCTION DE PRÉDICTION L2
# =================================================================
function predictions_L2(d, T; method=:conjugate_gradient, tol=1e-12)
    n = length(d)
    
    # Construction de la matrice de conception
    X = hcat(
        ones(n),
        d,
        cos.(2π .* d ./ 365.25),
        sin.(2π .* d ./ 365.25),
        cos.(2π .* d ./ (10.7*365.25)),
        sin.(2π .* d ./ (10.7*365.25))
    )
    
    if method == :qr
        # Méthode par factorisation QR
        Q, R = qr(X)
        β = R \ (Matrix(Q)' * T)
        
    elseif method == :conjugate_gradient
        # Méthode par gradient conjugué
        A = X' * X
        b = X' * T
        n_params = size(A, 1)
        
        β = zeros(n_params)
        r = b - A * β
        p = r
        rsold = r' * r
        
        for i in 1:1000 
            Ap = A * p
            α = rsold / (p' * Ap)
            β = β + α * p
            r = r - α * Ap
            rsnew = r' * r
            
            if sqrt(rsnew) < tol
                break
            end
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        end
        
    else
        error("Méthode non supportée: $method")
    end
    
    return β
end


# =================================================================
# 10. VISUALISATION DES RÉSULTATS DU MODÈLE L2
# =================================================================
gr()

# -----------------------------------------------------------------
# Graphique 1: Données et prédictions
# -----------------------------------------------------------------
p1 = plot(size=(800, 400), legend=:topleft)
scatter!(p1, d, T, 
         markersize=1, 
         alpha=0.3, 
         color=:lightblue, 
         label="Données brutes",
         xlabel="Jours depuis 1er janvier 1895",
         ylabel="Température (°F)",
         title="Températures moyennes journalières et modèle L2")

plot!(p1, d, T_hat_cg, 
      linewidth=2, 
      color=:red, 
      label="Modèle L2 (Gradient Conjugué)")

display(p1)
savefig(p1, "modele_L2_prediction.png")


# =================================================================
# 11. COMPOSANTES DU MODÈLE
# =================================================================

# -----------------------------------------------------------------
# Graphique 2: Composante saisonnière
# -----------------------------------------------------------------
saison_component = β_cg[3] .* cos.(2π .* d ./ 365.25) .+ β_cg[4] .* sin.(2π .* d ./ 365.25)

p3 = plot(size=(800, 400), legend=:topleft)
plot!(p3, d, saison_component,
      linewidth=2,
      color=:green,
      label="Composante saisonnière",
      xlabel="Jours depuis 1er janvier 1895",
      ylabel="Amplitude (°F)",
      title="Composante saisonnière du modèle")

display(p3)
savefig(p3, "modele_L2_saisonnier.png")

# -----------------------------------------------------------------
# Graphique 3: Composante linéaire 
# -----------------------------------------------------------------
lin_component = β_cg[1] .+ β_cg[2] .* d

p4 = plot(size=(600, 400), legend=:topleft)
plot!(p4, d, lin_component,
      linewidth=2,
      color=:blue,
      label="Tendance linéaire",
      xlabel="Jours depuis 1er janvier 1895",
      ylabel="Température (°F)",
      title="Composante linéaire (réchauffement)")

display(p4)
savefig(p4, "modele_L2_tendance_lineaire.png")

# -----------------------------------------------------------------
# Graphique 4: Composante solaire
# -----------------------------------------------------------------
solaire_component = β_cg[5] .* cos.(2π .* d ./ (10.7*365.25)) .+ β_cg[6] .* sin.(2π .* d ./ (10.7*365.25))

p5 = plot(size=(600, 400), legend=:topleft)
plot!(p5, d, solaire_component,
      linewidth=2,
      color=:orange,
      label="Cycle solaire",
      xlabel="Jours depuis 1er janvier 1895",
      ylabel="Amplitude (°F)",
      title="Composante solaire (10.7 ans)")

display(p5)
savefig(p5, "modele_L2_cycle_solaire.png")



println("\n" * "="^60)
println("TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS ET SAUVEGARDÉS")
println("="^60)
println("Fichiers créés:")
println("- modele_L2_prediction.png")
println("- modele_L2_saisonnier.png")
println("- modele_L2_tendance_lineaire.png")
println("- modele_L2_cycle_solaire.png")

println("="^60)