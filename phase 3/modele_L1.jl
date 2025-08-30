
# ===========================================================================================================================================
# 1. Librairies requises
# ===========================================================================================================================================
import Pkg
Pkg.add("GLPK")
Pkg.add("JuMP")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Printf")
using JuMP, GLPK, CSV, DataFrames, Printf

#GLPK (GNU Linear Programming Kit) : 
# un solveur open source qui permet de résoudre des problèmes de programmation linéaire (PL).

# ===========================================================================================================================================
# 2. Chargement des données
# ===========================================================================================================================================
# Adapter le chemin ci_dessous au chemin local de destination du dossier
data = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame)
# data = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/GHCND_USW00023183_clean.csv", DataFrame)
day = data.day
T = data.avg_temp
#T = data.value
n = length(day)

# ===========================================================================================================================================
# 3. Modèle d’optimisation
# ===========================================================================================================================================
model = Model(GLPK.Optimizer)

@variables(model, begin
    x[1:6]              # Coefficients x0 à x5
    dev[1:n] >= 0       # Variables pour les valeurs absolues
end)

@objective(model, Min, sum(dev))

# Contraintes LAD :
@constraint(model, [i in 1:n],
    x[1] + x[2]*day[i] +
    x[3]*cos(2π*day[i]/365.25) +
    x[4]*sin(2π*day[i]/365.25) +
    x[5]*cos(2π*day[i]/(10.7*365.25)) +
    x[6]*sin(2π*day[i]/(10.7*365.25)) - T[i] <= dev[i])

@constraint(model, [i in 1:n],
    -(x[1] + x[2]*day[i] +
    x[3]*cos(2π*day[i]/365.25) +
    x[4]*sin(2π*day[i]/365.25) +
    x[5]*cos(2π*day[i]/(10.7*365.25)) +
    x[6]*sin(2π*day[i]/(10.7*365.25)) - T[i]) <= dev[i])

# ===========================================================================================================================================
# 4. Résolution
# ===========================================================================================================================================
optimize!(model)
coeffs = value.(x)

# ===========================================================================================================================================
# 5. Affichage des résultats
# ===========================================================================================================================================
@printf("\nRésultats de la régression LAD (norme L1)\n")
@printf("------------------------------------------\n")
@printf("x0 (température moyenne de base)    = %.3f °F\n", coeffs[1])
@printf("x1 (pente de réchauffement local)   = %.6f °F/jour  → %.2f °F/siècle\n",
    coeffs[2], coeffs[2]*365.25*100)
@printf("x2, x3 (effet saisonnier)           = %.3f , %.3f\n", coeffs[3], coeffs[4])
@printf("x4, x5 (effet cycle solaire)        = %.3f , %.3f\n", coeffs[5], coeffs[6])

amplitude_saison = sqrt(coeffs[3]^2 + coeffs[4]^2)
amplitude_solaire = sqrt(coeffs[5]^2 + coeffs[6]^2)

@printf("\nAmplitude saisonnière               = %.3f °F\n", amplitude_saison)
@printf("Amplitude du cycle solaire          = %.3f °F\n", amplitude_solaire)

#############################################################################################################################################

# Fonction de Prédiction L1
function predictions_L1(d, T)
    n = length(d)
    
    X = hcat(
        ones(n),
        d,
        cos.(2π .* d ./ 365.25),
        sin.(2π .* d ./ 365.25),
        cos.(2π .* d ./ (10.7*365.25)),
        sin.(2π .* d ./ (10.7*365.25))
    )
    
    model = Model(GLPK.Optimizer)
    @variable(model, x[1:6])
    @variable(model, residuals[1:n])
    @objective(model, Min, sum(residuals))
    
    for i in 1:n
        pred = sum(X[i,j] * x[j] for j in 1:6)
        @constraint(model, pred - T[i] ≤ residuals[i])
        @constraint(model, T[i] - pred ≤ residuals[i])
    end
    
    optimize!(model)
    return value.(x)
end


if termination_status(model) == MOI.OPTIMAL
    β_L1 = value.(x)
    T_L1 = [sum([β_L1[1], 
                 β_L1[2]*day[i], 
                 β_L1[3]*cos(2π*day[i]/365.25), 
                 β_L1[4]*sin(2π*day[i]/365.25),
                 β_L1[5]*cos(2π*day[i]/(10.7*365.25)),
                 β_L1[6]*sin(2π*day[i]/(10.7*365.25))]) for i in 1:n]
    
    residuals_L1 = T - T_L1
    abs_error_L1 = abs.(residuals_L1)
    
    println("Solution L1 (LAD) trouvée!")
    
    # =================================================================
    # VISUALISATION 1: Données vs Prédictions L1
    # =================================================================
    p1 = plot(size=(1000, 600), legend=:topleft)
    plot!(p1, day, T, 
        label="Données réelles", 
        linewidth=0.5, 
        color=:blue,
        alpha=0.6)

    plot!(p1, day, T_L1, 
        label="Prédictions L1", 
        inewidth=2.5, 
        color=:red)
    
    xlabel!("Jours depuis la date de référence")
    ylabel!("Température (°F)")
    title!("Modèle L1 (LAD) - Données vs Prédictions")
    
    display(p1)
    savefig(p1, "modele_L1_predictions.png")
 
    # =================================================================
    # VISUALISATION 2: Composantes du modèle L1
    # =================================================================
    # Composante tendance linéaire
    tend_linaire = β_L1[1] .+ β_L1[2] .* day
    
    # Composante saisonnière
    comp_saisonniere = β_L1[3] .* cos.(2π .* day ./ 365.25) .+ 
                        β_L1[4] .* sin.(2π .* day ./ 365.25)
    
    # Composante solaire
    comp_solaire = β_L1[5] .* cos.(2π .* day ./ (10.7*365.25)) .+ 
                     β_L1[6] .* sin.(2π .* day ./ (10.7*365.25))
    
    p2 = plot(layout=(3,1), size=(1000, 800), legend=:topleft)
    
    # Tendance linéaire
    plot!(p2[1], day, tend_linaire,
          label="Tendance linéaire",
          linewidth=2,
          color=:red)
    title!(p2[1], "Composante tendance linéaire")
    ylabel!(p2[1], "Température (°F)")
    
    # Composante saisonnière
    plot!(p2[2], day, comp_saisonniere,
          label="Composante saisonnière",
          linewidth=2,
          color=:orange)
    title!(p2[2], "Composante saisonnière (1 an)")
    ylabel!(p2[2], "Amplitude (°F)")
    
    # Composante solaire
    plot!(p2[3], day, comp_solaire,
          label="Composante solaire",
          linewidth=2,
          color=:blue)
    title!(p2[3], "Composante cycle solaire (10.7 ans)")
    xlabel!(p2[3], "Jours depuis la date de référence")
    ylabel!(p2[3], "Amplitude (°F)")
    
    display(p2)
    savefig(p2, "modele_L1_composantes.png")   
    
else
    println("Échec de la résolution du modèle L1!")
    println("Statut de terminaison: ", termination_status(model))
end