
function calculate_humidex(T_F, dewpt_F)
    T_K = (T_F - 32) * 5/9 + 273.15
    D_K = (dewpt_F - 32) * 5/9 + 273.15
    H = T_F + 6.11 * exp(5417.7530 * (1/273.15 - 1/D_K)) - 10
    return H
end


import Pkg
Pkg.add("LinearAlgebra")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Printf")
Pkg.add("Statistics")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, Statistics, Printf, Plots

# =========================================================================================================================
# 1. Chargement et préparation des données
# =========================================================================================================================
function load_data(filepath)
    df = CSV.read(filepath, DataFrame)
    d = df.day
    T = df.avg_temp
    dewpt = hasproperty(df, :dew_point) ? df.dew_point : nothing
    H = isnothing(dewpt) ? nothing : calculate_humidex.(T, dewpt)
    return d, T, dewpt, H
end

# =========================================================================================================================
# 2. Fonction de régression L2 générique
# =========================================================================================================================
function run_L2_regression(d, Y; model_name="Température")
    n = length(d)
    X = hcat(
        ones(n),
        d,
        cos.(2π .* d ./ 365.25),
        sin.(2π .* d ./ 365.25),
        cos.(2π .* d ./ (10.7*365.25)),
        sin.(2π .* d ./ (10.7*365.25))
    )
    
    β = X \ Y
    Y_pred = X * β
    
    # Calcul des métriques
    residuals = Y - Y_pred
    rmse = sqrt(mean(residuals.^2))
    r_squared = 1 - sum(residuals.^2)/sum((Y .- mean(Y)).^2)
    
    # Affichage des résultats
    @printf("\n=== Régression L2 pour %s ===\n", model_name)
    @printf("--------------------------------\n")
    @printf("Coefficients:\n")
    @printf("x0 (constante)       = %8.3f °F\n", β[1])
    @printf("x1 (tendance)        = %8.6f °F/j → %.2f °F/siècle\n", β[2], β[2]*365.25*100)
    
    # Calcul des amplitudes
    amp_saison = sqrt(β[3]^2 + β[4]^2)
    amp_solaire = sqrt(β[5]^2 + β[6]^2)
    @printf("\nAmplitude saisonnière  = %8.3f °F\n", amp_saison)
    @printf("Amplitude solaire      = %8.3f °F\n", amp_solaire)
     
    return β, Y_pred, X
end

# =========================================================================================================================
# 3. Exécution principale
# =========================================================================================================================

# Adapter le chemin ci_dessous au chemin local de destination du dossier
d, T, dewpt, H = load_data("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv")

# Modèle température seule
β_temp, T_pred, X_temp = run_L2_regression(d, T, model_name="Température seule")

# Initialisation des variables pour humidex
β_humidex = nothing
H_pred = nothing
X_humidex = nothing

# Modèle humidex si données disponibles
if !isnothing(H)
    β_humidex, H_pred, X_humidex = run_L2_regression(d, H, model_name="Humidex")
    
    # Comparaison des tendances
    @printf("\n=== Comparaison des tendances ===\n")
    @printf("Température: %6.2f °F/siècle\n", β_temp[2]*365.25*100)
    @printf("Humidex:     %6.2f °F/siècle\n", β_humidex[2]*365.25*100)
    @printf("Différence:  %6.2f °F/siècle\n", (β_humidex[2]-β_temp[2])*365.25*100)
end

# =========================================================================================================================
# 4. Visualisation
# =========================================================================================================================
function create_plots(d, T, H, T_pred, H_pred)
    p1 = plot(d, T, label="Température réelle", color=:blue, alpha=0.5)
    plot!(d, T_pred, label="Modèle_L2 température", color=:red, lw=2)
    
    if !isnothing(H) && !isnothing(H_pred)
        plot!(d, H, label="Humidex réel", color=:green, alpha=0.5)
        plot!(d, H_pred, label="Modèle_L2 humidex", color=:red, lw=2)
    end
    
    title!("Régression L2 - Données vs Modèle")
    xlabel!("Jours depuis 01/01/1955")
    ylabel!("Valeur (°F)")
    
    if !isnothing(H) && !isnothing(H_pred)
        p2 = plot(d, H - T, label="Écart humidex-température", color=:purple)
        hline!([mean(H - T)], label="Moyenne", linestyle=:dash)
        title!("Écart humidex - température")
        ylabel!("Δ°F")
        
        return plot(p1, p2, layout=(2,1), size=(800,600))
    else
        return p1
    end
end


plot_result = if !isnothing(H) && !isnothing(H_pred)
    create_plots(d, T, H, T_pred, H_pred)
else
    create_plots(d, T, nothing, T_pred, nothing)
end

display(plot_result)