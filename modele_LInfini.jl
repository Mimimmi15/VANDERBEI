
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

# Chargez les fonctions depuis les fichiers
include("modele_L1.jl")  
include("modele_L2.jl")  

# ========================================================================================================================================================================
# 1. Chargement des données
# ========================================================================================================================================================================

# Adapter le chemin ci_dessous au chemin local de destination du dossier
df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame)
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

# Modèle 𝐿_inf - 
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
    
    # Calcul des dates caractéristiques
    #phase_saison = atan(β_Linf[4], β_Linf[3]) * 365.25/(2π)
    #@printf("\nJour le plus froid (prédit)         = %.0f janvier\n", 22 + phase_saison)
else
    @printf("\nÉchec de la résolution du modèle L_inf!\n")
end

# Vérification que la solution existe
if termination_status(model_Linf) == MOI.OPTIMAL
    β_Linf = value.(x)
    T_Linf = X * β_Linf
    
    println("Solution L_inf trouvée avec succès!")
    
    # Visualisation
    plot(d, T, label="Données réelles", alpha=0.5)
    plot!(d, T_L2, label="L2 (moindres carrés)", lw=2)
    plot!(d, T_L1, label="L1 (LAD)", lw=2)
    plot!(d, T_Linf, label="L_inf (erreur max)", lw=2)
    xlabel!("Jours")
    ylabel!("Température (°F)")
    title!("Comparaison des modèles")
else
    println("Échec de la résolution du modèle L_inf!")
    println("Statut de terminaison: ", termination_status(model_Linf))
end

