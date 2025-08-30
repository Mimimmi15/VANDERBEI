
import Pkg
Pkg.add("LinearAlgebra")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Printf")
Pkg.add("Statistics")
Pkg.add("Plots")
using CSV, DataFrames, LinearAlgebra, Statistics, Printf, Plots


# =================================================================
# FONCTION CALCUL HUMIDEX
# =================================================================
function calculate_humidex(T_F, dewpt_F)
    T_K = (T_F - 32) * 5/9 + 273.15
    D_K = (dewpt_F - 32) * 5/9 + 273.15
    H = T_F + 6.11 * exp(5417.7530 * (1/273.15 - 1/D_K)) - 10
    return H
end

# =================================================================
# FONCTION RÉGRESSION L2 AVEC GRADIENT CONJUGUÉ
# =================================================================
function run_L2_regression_conjugate_gradient(d, Y; model_name="Température", tol=1e-12)
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
    
    # Gradient conjugué
    A = X' * X
    b = X' * Y
    
    β = zeros(6)
    r = b - A * β
    p = r
    rsold = r' * r
    
    # Algorithme du gradient conjugué
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
    
    Y_pred = X * β
    
    
    # Calcul des amplitudes
    amp_saison = sqrt(β[3]^2 + β[4]^2)
    amp_solaire = sqrt(β[5]^2 + β[6]^2)
    
    # Affichage des résultats
    println("="^65)
    @printf("RÉGRESSION L2 AVEC GRADIENT CONJUGUÉ - %s\n", uppercase(model_name))
    println("="^65)
    
    @printf("x0 (constante)       = %10.3f °F\n", β[1])
    @printf("x1 (tendance)        = %10.6f °F/j → %8.2f °F/siècle\n", 
            β[2], β[2] * 365.25 * 100)
    @printf("x2 (saison cos)      = %10.3f\n", β[3])
    @printf("x3 (saison sin)      = %10.3f\n", β[4])
    @printf("x4 (solaire cos)     = %10.3f\n", β[5])
    @printf("x5 (solaire sin)     = %10.3f\n", β[6])
    
    println("\n" * "-"^45)
    @printf("Amplitude saisonnière  = %10.3f °F\n", amp_saison)
    @printf("Amplitude solaire      = %10.3f °F\n", amp_solaire)
    println("="^65)
    
    return β, Y_pred, X
end

# =================================================================
# CHARGEMENT DES DONNÉES AVEC HUMIDEX
# =================================================================
function load_data_with_humidex(filepath)
    df = CSV.read(filepath, DataFrame)
    d = df.day
    T = df.avg_temp
    
    # Vérification de la présence du point de rosée
    if hasproperty(df, :dew_point) && !all(ismissing.(df.dew_point))
        dewpt = df.dew_point
        H = calculate_humidex.(T, dewpt)
        @printf("Données humidex disponibles (%d points)\n", length(H))
        return d, T, dewpt, H
    else
        @printf("Données point de rosée non disponibles\n")
        return d, T, nothing, nothing
    end
end

# =================================================================
# EXÉCUTION
# =================================================================
function main_humidex_analysis()
    
    filepath= "C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/temperatures_clean.csv"  # à adapter selon l'utilisateur
    # filepath = "C:/Users/fogue/Downloads/VANDERBEI-phase2/VANDERBEI-phase2/GHCND_USW00023183_clean.csv"
    d, T, dewpt, H = load_data_with_humidex(filepath)
    
    # Régression pour la température
    println("\n" * "ANALYSE TEMPÉRATURE SEULE")
    β_temp, T_pred, X = run_L2_regression_conjugate_gradient(d, T, model_name="Température")
    
    # Régression pour l'humidex (si disponible)
    if !isnothing(H)
        println("\n" * "ANALYSE HUMIDEX")
        β_humidex, H_pred, X_humidex = run_L2_regression_conjugate_gradient(d, H, model_name="Humidex")
        
        # Comparaison des tendances
        println("\n" * "="^65)
        println("COMPARAISON TEMPÉRATURE vs HUMIDEX")
        println("="^65)
        
        trend_temp = β_temp[2] * 365.25 * 100
        trend_humidex = β_humidex[2] * 365.25 * 100
        trend_diff = trend_humidex - trend_temp
        
        @printf("Tendance température: %8.2f °F/siècle\n", trend_temp)
        @printf("Tendance humidex:     %8.2f °F/siècle\n", trend_humidex)
        @printf("Différence:           %8.2f °F/siècle\n", trend_diff)
        @printf("Écart relatif:        %8.1f%%\n", (trend_diff / trend_temp) * 100)
        
        # Comparaison des amplitudes saisonnières
        amp_temp = sqrt(β_temp[3]^2 + β_temp[4]^2)
        amp_humidex = sqrt(β_humidex[3]^2 + β_humidex[4]^2)
        amp_diff = amp_humidex - amp_temp
        
        println("\n" * "-"^45)
        @printf("Amplitude saison (temp): %8.3f °F\n", amp_temp)
        @printf("Amplitude saison (hum):  %8.3f °F\n", amp_humidex)
        @printf("Différence:              %8.3f °F\n", amp_diff)
        @printf("Écart relatif:           %8.1f%%\n", (amp_diff / amp_temp) * 100)
        
        println("="^65)
        
        return β_temp, β_humidex, T_pred, H_pred
    end
    
    return β_temp, nothing, T_pred, nothing
end

# =================================================================
# FONCTION DE PRÉDICTION
# =================================================================
function predictions_L2_humidex(d, Y; method=:conjugate_gradient)
    n = length(d)
    
    X = hcat(
        ones(n),
        d,
        cos.(2π .* d ./ 365.25),
        sin.(2π .* d ./ 365.25),
        cos.(2π .* d ./ (10.7*365.25)),
        sin.(2π .* d ./ (10.7*365.25))
    )
    
    if method == :conjugate_gradient
        # Gradient conjugué
        A = X' * X
        b = X' * Y
        
        β = zeros(6)
        r = b - A * β
        p = r
        rsold = r' * r
        
        for i in 1:1000
            Ap = A * p
            α = rsold / (p' * Ap)
            β = β + α * p
            r = r - α * Ap
            rsnew = r' * r
            
            if sqrt(rsnew) < 1e-12
                break
            end
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        end
        
    else
        β = X \ Y
    end
    
    return β
end
