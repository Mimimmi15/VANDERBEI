# =====================
# 1. Librairies requises
# =====================
import Pkg
Pkg.add("GLPK")
Pkg.add("JuMP")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Printf")
using JuMP, GLPK, CSV, DataFrames, Printf

#GLPK (GNU Linear Programming Kit) : 
# un solveur open source qui permet de résoudre des problèmes de programmation linéaire (PL) et en nombres entiers mixtes (PLNE).

# =====================
# 2. Chargement des données
# =====================
# besoin d'un fichier CSV contennant deux colonnes : day, avg_temp
data = CSV.read("temperatures.csv", DataFrame)
day = data.day
T = data.avg_temp
n = length(day)

# =====================
# 3. Modèle d’optimisation
# =====================
model = Model(GLPK.Optimizer)

@variables(model, begin
    x[1:6]              # Coefficients x0 à x5
    dev[1:n] >= 0       # Variables pour les valeurs absolues
end)

@objective(model, Min, sum(dev))

# Contraintes LAD : -dev[i] ≤ prédiction - T[i] ≤ dev[i]
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

# =====================
# 4. Résolution
# =====================
optimize!(model)
coeffs = value.(x)

# =====================
# 5. Affichage des résultats
# =====================
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
