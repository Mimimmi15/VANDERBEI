using Pkg
Pkg.add(["HTTP", "CSV", "DataFrames", "Dates", "JSON3","PyCall"])
using PyCall
pyimport("conda").activate("spyder-runtime")
using HTTP
using CSV
using DataFrames
using Dates: Date
using JSON3

# ========== CONFIGURATION ==========
const NOAA_API_KEY = "cBJbvohaShpTOUXRxRiwDxFqybZWCyoA"  
const BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
const HEADERS = Dict("token" => NOAA_API_KEY)

# ========== FONCTIONS ==========
function get_stations_list(output_file="noaa_stations.csv")
    stations_url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    
    try
        response = HTTP.get(stations_url)
        data = String(response.body)
        
        stations = DataFrame(
            ID = [strip(line[1:11]) for line in split(data, '\n') if length(line) ≥ 11],
            Latitude = [parse(Float64, line[12:20]) for line in split(data, '\n') if length(line) ≥ 20],
            Longitude = [parse(Float64, line[21:30]) for line in split(data, '\n') if length(line) ≥ 30],
            Name = [strip(line[41:71]) for line in split(data, '\n') if length(line) ≥ 71]
        )
        
        stations.ID = "GHCND:" .* stations.ID
        CSV.write(output_file, stations)
        return stations
    catch e
        @error "Erreur de téléchargement" exception=(e, catch_backtrace())
        return nothing
    end
end

function download_station_data(station_id; start_date="1955-01-01", end_date="2010-12-31", datatype="TAVG")
    try
        start_dt = Date(start_date)
        end_dt = Date(end_date)
    catch e
        @error "Format de date invalide" exception=e
        return nothing
    end

    all_data = DataFrame()
    current_date = start_dt
    
    while current_date <= end_dt
        block_end = min(current_date + Year(1) - Day(1), end_dt)
        
        params = Dict(
            "datasetid" => "GHCND",
            "stationid" => station_id,
            "startdate" => string(current_date),
            "enddate" => string(block_end),
            "datatypeid" => datatype,
            "units" => "standard",
            "limit" => 1000
        )
        
        try
            response = HTTP.get("$BASE_URL/data", headers=HEADERS, query=params)
            json_data = JSON3.read(response.body)
            
            if haskey(json_data, :results) && !isempty(json_data.results)
                chunk = DataFrame(json_data.results)
                all_data = vcat(all_data, chunk, cols=:union)
            end
        catch e
            @warn "Erreur API" date_range="$current_date-$block_end" exception=e
        end
        
        current_date = block_end + Day(1)
        sleep(1.5)  # Respect des limites de l'API
    end
    
    if !isempty(all_data)
        transform!(all_data, 
            :date => ByRow(d -> Date(d[1:10])) => :date,
            :value => ByRow(v -> v/10) => :temp_f
        )
    end
    
    return all_data
end

# ========== EXECUTION ==========
function main()
    # Création du dossier de sortie
    mkpath("data")
    
    # Stations cibles
    target_stations = [
        "GHCND:USW00013743",  # McGuire AFB
        "GHCND:USW00094728",  # NYC Central Park
        "GHCND:USW00023183"   # Los Angeles
    ]
    
    # Téléchargement séquentiel
    for station in target_stations
        @info "Début du traitement pour $station"
        data = download_station_data(station)
        
        if !isnothing(data) && !isempty(data)
            filename = "data/$(replace(station, ":" => "_")).csv"
            CSV.write(filename, data)
            @info "Données sauvegardées: $filename ($(nrow(data)) enregistrements)"
        else
            @warn "Aucune donnée obtenue pour $station"
        end
    end
end

# Point d'entrée principal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end