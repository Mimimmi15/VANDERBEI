"""
using Pkg
Pkg.add("CSV") 
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("JuMP")
Pkg.add("GLPK")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, JuMP, GLPK, Plots


include("Facteur_humidex_modele_L1.jl") 
include("Facteur_humidex_modele_L2.jl")  

# Adapter le chemin ci_dessous au chemin local de destination du dossier
df = CSV.read("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv", DataFrame) 
d = df.day
T = df.avg_temp
dewpt = df.dew_point
n = length(d)

# Calcul du humidex
H = calculate_humidex.(T, dewpt)

# Construction de la matrice X
X = hcat(
    ones(n),
    d,
    cos.(2π .* d ./ 365.25),
    sin.(2π .* d ./ 365.25),
    cos.(2π .* d ./ (10.7*365.25)),
    sin.(2π .* d ./ (10.7*365.25))
)


# Appel des modèles avec humidex
β_L1 = predictions_L1(d, H) 
β_L2 = predictions_L2(d, H)
T_L1 = X * β_L1
T_L2 = X * β_L2

# Modèle L_inf avec humidex
model_Linf = Model(GLPK.Optimizer)
@variable(model_Linf, x[1:6])
@variable(model_Linf, t >= 0)
@objective(model_Linf, Min, t)

for i in 1:n
    pred = sum(X[i,j] * x[j] for j in 1:6)
    @constraint(model_Linf, pred - H[i] <= t) 
    @constraint(model_Linf, -(pred - H[i]) <= t)
end

optimize!(model_Linf)


# =====================
# Affichage des résultats L_inf
# =====================
if termination_status(model_Linf) == MOI.OPTIMAL
    β_Linf = value.(x) 
    
    @printf("\nRésultats de la régression L∞ (minimax)\n")
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

    @printf("\n=== Résultats L∞ sur humidex ===\n")
    @printf("Erreur maximale minimisée       = %.3f °F\n", value(t))
    @printf("Date du pic d'erreur            = jour %.0f\n", d[findmax(abs.(H - X*β))[2]])
    # Calcul des dates caractéristiques
    #phase_saison = atan(β_Linf[4], β_Linf[3]) * 365.25/(2π)
    #@printf("\nJour le plus froid (prédit)         = %.0f janvier\n", 22 + phase_saison)
else
    @printf("\nÉchec de la résolution du modèle L∞!\n")
end

# Vérification que la solution existe
if termination_status(model_Linf) == MOI.OPTIMAL
    β_Linf = value.(x)
    T_Linf = X * β_Linf
    
    println("Solution L∞ trouvée avec succès!")
    
    # Visualisation
    plot(d, T, label="Données réelles", alpha=0.5)
    plot!(d, T_L2, label="L2 (moindres carrés)", lw=2)
    plot!(d, T_L1, label="L1 (LAD)", lw=2)
    plot!(d, T_Linf, label="L∞ (erreur max)", lw=2)
    xlabel!("Jours")
    ylabel!("Température (°F)")
    title!("Comparaison des modèles")
else
    println("Échec de la résolution du modèle L∞!")
    println("Statut de terminaison: ", termination_status(model_Linf))
end


##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################


using Pkg
Pkg.add("CSV") 
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("JuMP")
Pkg.add("GLPK")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, JuMP, GLPK, Plots

# =================================================================
# 1. Chargement des données et fonctions communes
# =================================================================
include("Facteur_humidex_modele_L1.jl") 
include("Facteur_humidex_modele_L2.jl")

function load_common_data(filepath)
    df = CSV.read(filepath, DataFrame)
    d = df.day
    T = df.avg_temp
    dewpt = hasproperty(df, :dew_point) ? df.dew_point : nothing
    H = isnothing(dewpt) ? nothing : calculate_humidex.(T, dewpt)
    return d, T, dewpt, H
end

# =================================================================
# 2. Modèle L∞ (version compatible)
# =================================================================
function run_Linf_regression(d, Y; model_name="Température")
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
    @variable(model, t >= 0)
    @objective(model, Min, t)
    
    for i in 1:n
        pred = sum(X[i,j] * x[j] for j in 1:6)
        @constraint(model, pred - Y[i] <= t)
        @constraint(model, -(pred - Y[i]) <= t)
    end
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        β = value.(x)
        Y_pred = X * β
        max_error = value(t)
        
        # Affichage standardisé
        @printf("\n=== Régression L_inf pour %s ===\n", model_name)
        @printf("--------------------------------\n")
        @printf("Coefficients:\n")
        @printf("x0 (constante)       = %8.3f °F\n", β[1])
        @printf("x1 (tendance)        = %8.6f °F/j → %.2f °F/siècle\n", 
                β[2], β[2]*365.25*100)
        
        amp_saison = sqrt(β[3]^2 + β[4]^2)
        amp_solaire = sqrt(β[5]^2 + β[6]^2)
        @printf("\nAmplitude saisonnière  = %8.3f °F\n", amp_saison)
        @printf("Amplitude solaire      = %8.3f °F\n", amp_solaire)
        @printf("\nErreur maximale        = %8.3f °F\n", max_error)
        
        return β, Y_pred
    else
        @printf("\nÉchec de la résolution pour %s\n", model_name)
        return nothing, nothing
    end
end

# =================================================================
# 3. Exécution principale
# =================================================================
d, T, dewpt, H = load_common_data("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv")

# Modèles température et humidex
β_temp_Linf, T_pred_Linf = run_Linf_regression(d, T, model_name="Température seule")

if !isnothing(H)
    β_humidex_Linf, H_pred_Linf = run_Linf_regression(d, H, model_name="Humidex")
    
    # Comparaison des tendances
    if !isnothing(β_temp_Linf) && !isnothing(β_humidex_Linf)
        @printf("\n=== Comparaison des tendances ===\n")
        @printf("Température: %6.2f °F/siècle\n", β_temp_Linf[2]*365.25*100)
        @printf("Humidex:     %6.2f °F/siècle\n", β_humidex_Linf[2]*365.25*100)
        @printf("Différence:  %6.2f °F/siècle\n", 
                (β_humidex_Linf[2]-β_temp_Linf[2])*365.25*100)
    end
end

# =================================================================
# 4. Visualisation unifiée
# =================================================================
function create_combined_plot(d, T, H, preds...)
    p = plot(d, T, label="Température réelle", color=:blue, alpha=0.5)
    
    if !isnothing(H)
        plot!(d, H, label="Humidex réel", color=:red, alpha=0.5)
    end
    
    # Ajout des prédictions (L1, L2, L_inf)
    colors = [:green, :purple, :orange]
    labels = ["L1 (LAD)", "L2 (MC)", "L_inf (Minimax)"]
    for (i, (pred, label)) in enumerate(zip(preds, labels))
        if !isnothing(pred)
            plot!(d, pred, label=label, color=colors[i], lw=2)
        end
    end
    
    title!("Comparaison des modèles")
    xlabel!("Jours depuis 01/01/1955")
    ylabel!("Valeur (°F)")
    return p
end

# Appel des autres modèles pour comparaison
β_L1, T_L1 = predictions_L1(d, T), nothing
β_L2, T_L2 = predictions_L2(d, T), nothing

if !isnothing(H)
    β_L1, T_L1 = predictions_L1(d, H), X*β_L1
    β_L2, T_L2 = predictions_L2(d, H), X*β_L2
end

plot_result = create_combined_plot(d, T, H, T_L1, T_L2, T_pred_Linf)
display(plot_result)

"""




using Pkg
Pkg.add("CSV") 
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("JuMP")
Pkg.add("GLPK")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, JuMP, GLPK, Plots

# =================================================================
# 1. Chargement des données et fonctions
# =================================================================
function calculate_humidex(T_F, dewpt_F)
    T_K = (T_F - 32) * 5/9 + 273.15
    D_K = (dewpt_F - 32) * 5/9 + 273.15
    H = T_F + 6.11 * exp(5417.7530 * (1/273.15 - 1/D_K)) - 10
    return H
end

function load_common_data(filepath)
    df = CSV.read(filepath, DataFrame)
    d = df.day
    T = df.avg_temp
    dewpt = hasproperty(df, :dew_point) ? df.dew_point : nothing
    H = isnothing(dewpt) ? nothing : calculate_humidex.(T, dewpt)
    return d, T, dewpt, H
end

# =================================================================
# 2. Modèle L_inf
# =================================================================
function run_Linf_regression(d, Y; model_name="Température")
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
    @variable(model, t >= 0)
    @objective(model, Min, t)
    
    for i in 1:n
        pred = sum(X[i,j] * x[j] for j in 1:6)
        @constraint(model, pred - Y[i] <= t)
        @constraint(model, -(pred - Y[i]) <= t)
    end
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        β = value.(x)
        Y_pred = X * β
        max_error = value(t)
        
        # Affichage standardisé
        @printf("\n=== Régression L_inf pour %s ===\n", model_name)
        @printf("--------------------------------\n")
        @printf("Coefficients:\n")
        @printf("x0 (constante)       = %8.3f °F\n", β[1])
        @printf("x1 (tendance)        = %8.6f °F/j → %.2f °F/siècle\n", 
                β[2], β[2]*365.25*100)
        
        amp_saison = sqrt(β[3]^2 + β[4]^2)
        amp_solaire = sqrt(β[5]^2 + β[6]^2)
        @printf("\nAmplitude saisonnière  = %8.3f °F\n", amp_saison)
        @printf("Amplitude solaire      = %8.3f °F\n", amp_solaire)
        @printf("\nErreur maximale        = %8.3f °F\n", max_error)
        
        return β, Y_pred, X
    else
        @printf("\nÉchec de la résolution pour %s\n", model_name)
        return nothing, nothing, nothing
    end
end

# =================================================================
# 3. Exécution principale
# =================================================================
d, T, dewpt, H = load_common_data("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv")

# Modèle L_inf température
#β_temp_Linf, T_pred_Linf, X = run_Linf_regression(d, T, model_name="Température seule")

# Initialisation des variables
β_temp, T_pred = nothing, nothing
β_humidex, H_pred = nothing, nothing

# Modèle température
if !isnothing(T)
    β_temp, T_pred, X = run_Linf_regression(d, T, model_name="Température")
end

# Modèle humidex si données disponibles
if !isnothing(H)
    β_humidex, H_pred, _ = run_Linf_regression(d, H, model_name="Humidex")
end

# Comparaison si les deux modèles ont réussi
if !isnothing(β_temp) && !isnothing(β_humidex)
    @printf("\n=== Comparaison ===\n")
    @printf("Différence de tendance: %.2f °F/siècle\n",
            (β_humidex[2] - β_temp[2]) * 365.25 * 100)
end

# Visualisation sécurisée
function create_plot(d, T, H, T_pred, H_pred)
    p = plot(d, T, label="Température réelle", color=:blue, alpha=0.5)
    
    if !isnothing(T_pred)
        plot!(d, T_pred, label="Modèle L_inf température", color=:green, lw=2)
    end
    
    if !isnothing(H) && !isnothing(H_pred)
        plot!(d, H, label="Humidex réel", color=:red, alpha=0.5)
        plot!(d, H_pred, label="Modèle L_inf humidex", color=:red, lw=2)
    end
    
    title!("Régression L_inf")
    xlabel!("Jours depuis 01/01/1955")
    ylabel!("Valeur (°F)")
    return p
end

# Appel sécurisé de la visualisation
plot_result = create_plot(d, T, H, T_pred, H_pred)
display(plot_result)
