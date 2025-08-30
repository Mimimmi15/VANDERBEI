
# Reproduire le modèle de Vanderbei avec une régression en norme 𝐿_inf
#revient à minimiser l’erreur maximale absolue entre le modèle et les données, 
#ce qui donne aussi un problème de programmation linéaire (comme pour la norme 𝐿1)

using Pkg
Pkg.add("CSV") 
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("JuMP")
Pkg.add("GLPK")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, JuMP, GLPK, Plots

include("modele_L1.jl")  
include("modele_L2.jl")  

# ========================================================================================================================================================================
# 1. Chargement des données
# ========================================================================================================================================================================

# Adapter le chemin ci_dessous au chemin local de destination du dossier
df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame)
#df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/GHCND_USW00023183_clean.csv", DataFrame)
d = df.day
T = df.avg_temp
n = length(d)

# ========================================================================================================================================================================
# 2. Construction des prédicteurs
# ========================================================================================================================================================================
X = hcat(
    ones(n),
    d,
    cos.(2π .* d ./ 365.25),
    sin.(2π .* d ./ 365.25),
    cos.(2π .* d ./ (10.7*365.25)),
    sin.(2π .* d ./ (10.7*365.25))
)

# ========================================================================================================================================================================
# 3. Résolution du modèle L_inf
# ========================================================================================================================================================================
β_L1 = predictions_L1(d, T)
β_L2 = predictions_L2(d, T)
T_L1 = X * β_L1
T_L2 = X * β_L2

# Modèle 𝐿_inf 
println("Résolution du modèle L_inf...")
model_Linf = Model(GLPK.Optimizer)

# Déclaration des variables
@variable(model_Linf, x[1:6]) 
@variable(model_Linf, t >= 0)

# Définition de l'objectif
@objective(model_Linf, Min, t)

# Contraintes
for i in 1:n
    pred = sum(X[i,j] * x[j] for j in 1:6)
    @constraint(model_Linf, pred - T[i] <= t)
    @constraint(model_Linf, T[i] - pred <= t)
end

optimize!(model_Linf)

# ========================================================================================================================================================================
# Affichage des résultats L_inf
# ========================================================================================================================================================================
if termination_status(model_Linf) == MOI.OPTIMAL
    β_Linf = value.(x) 
    
    @printf("\nRésultats de la régression L_inf (minimax)\n")
    @printf("------------------------------------------\n")
    @printf("x0 (température moyenne de base)    = %.3f °F\n", β_Linf[1])
    @printf("x1 (pente de réchauffement local)   = %.6f °F/jour → %.2f °F/siècle\n",
            β_Linf[2], β_Linf[2]*365.25*100)
    @printf("x2, x3 (effet saisonnier)           = %.3f, %.3f\n", β_Linf[3], β_Linf[4])
    @printf("x4, x5 (effet cycle solaire)        = %.3f, %.3f\n", β_Linf[5], β_Linf[6])
    
    amplitude_saison = sqrt(β_Linf[3]^2 + β_Linf[4]^2)
    amplitude_solaire = sqrt(β_Linf[5]^2 + β_Linf[6]^2)
    erreur_max = value(t) 
    
    @printf("\nAmplitude saisonnière               = %.3f °F\n", amplitude_saison)
    @printf("Amplitude du cycle solaire          = %.3f °F\n", amplitude_solaire)
    @printf("Erreur maximale minimisée           = %.3f °F\n", erreur_max)
    
else
    @printf("\nÉchec de la résolution du modèle L_inf!\n")
end


if termination_status(model_Linf) == MOI.OPTIMAL
    β_Linf = value.(x)
    T_Linf = X * β_Linf
    
    println("Solution L_inf trouvée!")
    
    # =================================================================
    # VISUALISATION 1: Comparaison des modèles
    # =================================================================
    p1 = plot(size=(1000, 600), legend=:topleft)
    plot!(p1, d, T, label="Données réelles", linewidth=2, color=:orange)
    plot!(p1, d, T_L2, label="L2 (moindres carrés)", linewidth=2, color=:blue)
    plot!(p1, d, T_L1, label="L1 (LAD)", linewidth=2, color=:red)
    plot!(p1, d, T_Linf, label="L∞ (minimax)", linewidth=2, color=:green)
    xlabel!("Jours depuis la date de référence")
    ylabel!("Température (°F)")
    title!("Comparaison des trois modèles de régression")
    
    display(p1)
    savefig(p1, "comparaison_modeles.png")

    # =================================================================
    # VISUALISATION 2: Données vs Prédictions
    # =================================================================
    p2 = plot(size=(1000, 600), legend=:topleft)
    plot!(p2, d, T, 
             label="Données réelles",
             linewidth=2, 
             color=:green)
    
    plot!(p2, d, T_Linf, 
          label="Prédictions L∞ (minimax)", 
          linewidth=2, 
          color=:red)
    
    xlabel!("Jours depuis la date de référence")
    ylabel!("Température (°F)")
    title!("Modèle L∞ - Données vs Prédictions")
    
    display(p2)
    savefig(p2, "modele_Linf_predictions.png")
    

    # =================================================================
    # VISUALISATION 3: Composantes du modèle L_inf
    # =================================================================
    # Composante tendance linéaire
    tendance_lineaire = β_Linf[1] .+ β_Linf[2] .* d
    
    # Composante saisonnière
    comp_saisonniere = β_Linf[3] .* cos.(2π .* d ./ 365.25) .+ 
                        β_Linf[4] .* sin.(2π .* d ./ 365.25)
    
    # Composante solaire
    comp_solaire = β_Linf[5] .* cos.(2π .* d ./ (10.7*365.25)) .+ 
                     β_Linf[6] .* sin.(2π .* d ./ (10.7*365.25))
    
    p3 = plot(layout=(3,1), size=(1000, 800), legend=:topleft)
    
    # Tendance linéaire
    plot!(p3[1], d, tendance_lineaire,
          label="Tendance linéaire",
          linewidth=2,
          color=:blue)
    title!(p3[1], "Composante tendance linéaire")
    ylabel!(p3[1], "Température (°F)")
    
    # Composante saisonnière
    plot!(p3[2], d, comp_saisonniere,
          label="Composante saisonnière",
          linewidth=2,
          color=:green)
    title!(p3[2], "Composante saisonnière (1 an)")
    ylabel!(p3[2], "Amplitude (°F)")
    
    # Composante solaire
    plot!(p3[3], d, comp_solaire,
          label="Composante solaire",
          linewidth=2,
          color=:orange)
    title!(p3[3], "Composante cycle solaire (10.7 ans)")
    xlabel!(p3[3], "Jours depuis la date de référence")
    ylabel!(p3[3], "Amplitude (°F)")
    
    display(p3)
    savefig(p3, "modele_Linf_composantes.png")

else
    println("Échec de la résolution du modèle L_inf!")
    println("Statut de terminaison: ", termination_status(model_Linf))
end
