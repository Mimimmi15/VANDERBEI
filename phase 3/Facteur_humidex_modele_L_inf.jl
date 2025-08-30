
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
        
        # Affichage
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
d, T, dewpt, H = load_common_data("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv")  # à adapter selon l'utilisateur
#d, T, dewpt, H = load_common_data("C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/GHCND_USW00023183_clean.csv")

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

# Visualisation 
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

plot_result = create_plot(d, T, H, T_pred, H_pred)
display(plot_result)
