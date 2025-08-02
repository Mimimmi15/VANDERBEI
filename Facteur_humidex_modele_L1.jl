
function calculate_humidex(T_F, dewpt_F)
    T_K = (T_F - 32) * 5/9 + 273.15
    D_K = (dewpt_F - 32) * 5/9 + 273.15
    H = T_F + 6.11 * exp(5417.7530 * (1/273.15 - 1/D_K)) - 10
    return H
end

# =========================================================================================================================
# 1. Librairies requises
# =========================================================================================================================
import Pkg
Pkg.add("GLPK")
Pkg.add("JuMP")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Printf")
Pkg.add("Statistics")
Pkg.add("Plots")
using JuMP, GLPK, CSV, DataFrames, Printf, Statistics, Plots

# =========================================================================================================================
# 2. Chargement des données
# =========================================================================================================================

# Adapter le chemin ci_dessous au chemin local de destination du dossier
data = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame)
day = data.day
T = data.avg_temp
dewpt = hasproperty(data, :dew_point) ? data.dew_point : nothing
n = length(day)

# Calcul du humidex si les données de point de rosée sont disponibles
H = isnothing(dewpt) ? nothing : calculate_humidex.(T, dewpt)

# =========================================================================================================================
# 3. Modèle d'optimisation (version humidex)
# =========================================================================================================================
function run_L1_regression(d, Y; model_name="Température")
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
    @variable(model, residuals[1:n] >= 0)
    @objective(model, Min, sum(residuals))
    
    for i in 1:n
        pred = sum(X[i,j] * x[j] for j in 1:6)
        @constraint(model, pred - Y[i] ≤ residuals[i])
        @constraint(model, Y[i] - pred ≤ residuals[i])
    end
    
    optimize!(model)
    β = value.(x)
    Y_pred = X * β
    
    # Affichage des résultats
    @printf("\nRésultats de la régression LAD pour %s\n", model_name)
    @printf("------------------------------------------\n")
    @printf("Valeur de base (x0)            = %.3f °F\n", β[1])
    @printf("Tendance linéaire (x1)         = %.6f °F/j → %.2f °F/siècle\n",
            β[2], β[2]*365.25*100)
    
    # Calcul des amplitudes
    amp_saison = sqrt(β[3]^2 + β[4]^2)
    amp_solaire = sqrt(β[5]^2 + β[6]^2)
    @printf("Amplitude saisonnière          = %.3f °F\n", amp_saison)
    @printf("Amplitude cycle solaire        = %.3f °F\n", amp_solaire)
    
    # Métriques d'erreur
    errors = Y - Y_pred
    @printf("Erreur médiane absolue         = %.3f °F\n", median(abs.(errors)))
    @printf("Erreur maximale absolue        = %.3f °F\n", maximum(abs.(errors)))
    
    return β, Y_pred
end

# =========================================================================================================================
# 4. Exécution des modèles
# =========================================================================================================================
# Modèle température seule
β_temp, T_pred = run_L1_regression(day, T, model_name="Température seule")

# Modèle humidex si les données sont disponibles
if !isnothing(H)
    β_humidex, H_pred = run_L1_regression(day, H, model_name="Humidex")
    
    # Comparaison des tendances
    @printf("\nComparaison des tendances:\n")
    @printf("-> Température: %.2f °F/siècle\n", β_temp[2]*365.25*100)
    @printf("-> Humidex:     %.2f °F/siècle\n", β_humidex[2]*365.25*100)
    @printf("-> Différence:  %.2f °F/siècle\n", 
            (β_humidex[2] - β_temp[2])*365.25*100)
end

# =========================================================================================================================
# 5. Visualisation
# =========================================================================================================================

if !isnothing(H)
    p1 = plot(day, T, label="Température", color=:blue, alpha=0.7)
    plot!(day, H, label="Humidex", color=:red, alpha=0.7)
    plot!(day, T_pred, label="Modèle température", color=:blue)
    plot!(day, H_pred, label="Modèle humidex", color=:red)
    title!("Comparaison température et humidex")
    xlabel!("Jours")
    ylabel!("Valeur (°F)")
    
    p2 = plot(day, H - T, label="Écart humidex - température", color=:purple)
    hline!([mean(H - T)], label="Moyenne", linestyle=:dash)
    title!("Écart entre humidex et température")
    ylabel!("Δ°F")
    
    plot(p1, p2, layout=(2,1), size=(800,600))
end