using Pkg
Pkg.add("DataFrames")
Pkg.add("CSV")
using Dates
using DataFrames
using Statistics
using DelimitedFiles
using CSV


function load_temperatures(filename)
    # Lire toutes les lignes non vides
    lines = readlines(filename)
    non_empty_lines = filter(!isempty, lines)
    
    # Extraire uniquement les températures (dernière colonne)
    temperatures = Float64[]
    for line in non_empty_lines
        # Supprimer les espaces et splitter
        parts = split(strip(line))
        
        # Prendre le dernier élément comme température
        temp_str = last(parts)
        try
            push!(temperatures, parse(Float64, temp_str))
        catch e
            @warn "Impossible de parser la ligne: $line"
            push!(temperatures, missing)  # ou utiliser NaN
        end
    end
    return temperatures
end

# Chargement des températures
temperatures = load_temperatures("McGuireAFB.dat")

# Création des dates
start_date = Date(1955, 1, 1)
end_date = start_date + Day(length(temperatures) - 1)
dates = start_date:Dates.Day(1):end_date

# Création du DataFrame
df = DataFrame(
    date = collect(dates),
    temp_f = temperatures,
    days_since_1955 = [Dates.value(d - start_date) for d in dates]
)

# ----------------------------------------------
# Nettoyage des données manquantes
# ----------------------------------------------

# 1. Vérification des valeurs manquantes avant interpolation
missing_count = sum(ismissing.(df.temp_f))
println("\nNombre de valeurs manquantes avant interpolation : ", missing_count)

# 2. Interpolation (uniquement si nécessaire)
if missing_count > 0
    @info "Application de l'interpolation linéaire..."
    using Impute
    try
        df.temp_f = Impute.interp(df.temp_f)
        println("Valeurs manquantes après interpolation : ", sum(ismissing.(df.temp_f)))
    catch e
        @error "Échec de l'interpolation" exception=(e, catch_backtrace())
        # Alternative : remplissage avec la moyenne
        avg_temp = mean(skipmissing(df.temp_f))
        df.temp_f = coalesce.(df.temp_f, avg_temp)
    end
else
    @info "Aucune valeur manquante détectée - interpolation non nécessaire"
end

# ----------------------------------------------
# Sauvegarde des données nettoyées
# ----------------------------------------------

# 1. Création du répertoire de sortie si inexistant
output_dir = "results"
if !isdir(output_dir)
    mkdir(output_dir)
    @info "Répertoire $output_dir créé"
end

# 2. Sauvegarde avec options avancées
output_path = joinpath(output_dir, "temperatures_clean.csv")
try
    CSV.write(output_path, df;
        delim=',',
        missingstring="NA",
        dateformat="yyyy-mm-dd",
        quotestrings=true,
        bom=true
    )
    @info "Données sauvegardées avec succès dans $output_path"
catch e
    @error "Échec de la sauvegarde" exception=(e, catch_backtrace())
end

# Affichage des premières lignes
println("Premières lignes:")
show(first(df, 5), allcols=true)
println("\n")

# Vérifications
println("Nombre de jours : ", nrow(df))

# Température moyenne en 1955 (en excluant les missing)
df_1955 = filter(row -> year(row.date) == 1955 && !ismissing(row.temp_f), df)
println("Température moyenne en 1955 : ", mean(df_1955.temp_f))

# Aperçu des données
println("\nStatistiques descriptives:")
show(describe(df), allcols=true)

# ----------------------------------------------
# Version alternative avec Dates.dat
# ----------------------------------------------

function load_dates(filename)
    date_strings = String[]
    open(filename) do f
        for line in eachline(f)
            stripped = strip(line)
            if !isempty(stripped)
                push!(date_strings, stripped)
            end
        end
    end
    return date_strings
end

#=
# Décommenter cette section pour utiliser Dates.dat
date_strings = load_dates("Dates.dat")
dates_from_file = [Date(d[1:8], "yyyymmdd") for d in date_strings]

df_with_dates = DataFrame(
    date = dates_from_file,
    temp_f = temperatures[1:length(dates_from_file)],  # en cas de différence de longueur
    days_since_1955 = [Dates.value(d - Date(1955, 1, 1)) for d in dates_from_file]
)
=#

CSV.write("temperatures_clean.csv", df)
