# Reproduire le modèle de Vanderbei avec une régression en norme 𝐿∞
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

# =====================
# 1. Chargement des données
# =====================

df = CSV.read("temperatures.csv", DataFrame)
d = df.day
T = df.avg_temp
n = length(d)

# appel des modèles L1 et L2 existants
β_L1 = predictions_L1(d, T)
β_L2 = predictions_L2(d, T)


# =====================
# 2. Préparation des prédicteurs X (même base pour tous les modèles)
# =====================
X = [ones(n)                       # x0
     d                            # x1
     cos.(2π .* d ./ 365.25)      # x2
     sin.(2π .* d ./ 365.25)      # x3
     cos.(2π .* d ./ (10.7*365.25)) # x4
     sin.(2π .* d ./ (10.7*365.25))]'

X = X'
T_L1 = X * β_L1
T_L2 = X * β_L2

# =====================
# 3. Modèle L∞ (minimisation erreur max)
# =====================
model_Linf = Model(GLPK.Optimizer)
@variable(model_Linf, x_Linf[1:6])
@variable(model_Linf, t >= 0)
@objective(model_Linf, Min, t)

for i in 1:n
    pred = x_Linf[1] + x_Linf[2]*d[i] + x_Linf[3]*cos(2π*d[i]/365.25) +
           x_Linf[4]*sin(2π*d[i]/365.25) + x_Linf[5]*cos(2π*d[i]/(10.7*365.25)) +
           x_Linf[6]*sin(2π*d[i]/(10.7*365.25))
    @constraint(model_Linf, pred - T[i] <= t)
    @constraint(model_Linf, -(pred - T[i]) <= t)
end

optimize!(model_Linf)
β_Linf = value.(x_Linf)
T_Linf = X * β_Linf


# =====================
# 4. Visualisation avec graphe des trois normes ensembles 
# =====================

plot(d, T, label="Données réelles", color=:blue, lw=1, alpha=0.5)
plot!(d, T_L2, label="Modèle L2 (moindres carrés)", color=:red, lw=2)
plot!(d, T_L1, label="Modèle L1 (LAD)", color=:green, lw=2)
plot!(d, T_Linf, label="Modèle L∞ (erreur max)", color=:purple, lw=2)
xlabel!("Jour")
ylabel!("Température (°F)")
title!("Comparaison visuelle des modèles L1 / L2 / L∞")
