
using Pkg
Pkg.add(["HTTP", "CSV", "DataFrames", "Dates", "JSON3"])
using HTTP
using CSV
using DataFrames
using Dates: Date, Year, Day, year, month, day 
using JSON3

# ========== CONFIGURATION ==========
const NOAA_API_KEY = "cBJbvohaShpTOUXRxRiwDxFqybZWCyoA"  # à adapter selon l'utilisateur
const BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
const HEADERS = Dict("token" => NOAA_API_KEY)

# ========== FONCTIONS ==========
function get_stations_list(output_file="noaa_stations.csv")
    stations_url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    
    try
        println("Téléchargement de la liste des stations...")
        response = HTTP.get(stations_url)
        data = String(response.body)
        

        lines = filter(line -> length(line) ≥ 71, split(data, '\n', keepempty=false))
        
        stations = DataFrame(
            ID = [strip(line[1:11]) for line in lines],
            Latitude = [parse(Float64, strip(line[12:20])) for line in lines],
            Longitude = [parse(Float64, strip(line[21:30])) for line in lines],
            Elevation = [strip(line[31:37]) for line in lines],
            State = [strip(line[38:40]) for line in lines],
            Name = [strip(line[41:71]) for line in lines]
        )
        
        stations.ID = "GHCND:" .* stations.ID
        CSV.write(output_file, stations)
        println("Liste des stations sauvegardée dans $output_file ($(nrow(stations)) stations)")
        return stations
    catch e
        @error "Erreur de téléchargement" exception=(e, catch_backtrace())
        return nothing
    end
end

function download_station_data(station_id; start_date="1998-01-01", end_date="2010-12-31", datatype="TAVG")
    println("Téléchargement des données pour la station $station_id...")
    
    start_dt = Date(start_date)
    end_dt = Date(end_date)

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
            println("  Téléchargement $current_date au $block_end")
            response = HTTP.get("$BASE_URL/data"; headers=HEADERS, query=params)
            
            if response.status != 200
                @warn "Erreur HTTP: $(response.status)" station=station_id date_range="$current_date-$block_end"
                current_date = block_end + Day(1)
                continue
            end
            
            json_data = JSON3.read(response.body)
            
            if haskey(json_data, :results) && !isempty(json_data.results)
                chunk = DataFrame(json_data.results)
                all_data = vcat(all_data, chunk, cols=:union)
                println("    $(nrow(chunk)) enregistrements récupérés")
            else
                println("    Aucune donnée pour cette période")
            end
            
        catch e
            @warn "Erreur API" station=station_id date_range="$current_date-$block_end" exception=e
        end
        
        current_date = block_end + Day(1)
        sleep(1.5) 
    end
    
    if !isempty(all_data)

        select!(all_data, [:date, :datatype, :station, :value])
        transform!(all_data, 
            :date => ByRow(d -> Date(string(d)[1:10])) => :date,
            :value => ByRow(v -> v isa Number ? v/10 : missing) => :temp_f
        )
        dropmissing!(all_data)
        
        all_data[!, :year] = year.(all_data.date)
        all_data[!, :month] = month.(all_data.date)
        all_data[!, :day] = day.(all_data.date)
    end
    
    return all_data
end

# ========== EXECUTION ==========
function main()
    println("Début du programme de téléchargement NOAA...")
    
    # Création du dossier de sortie
    mkpath("data")
    
    stations = get_stations_list("data/noaa_stations.csv")
    

    target_stations = [
        "GHCND:USW00013743",  # Washington Reagan
        "GHCND:USW00094728",  # NYC Central Park
        "GHCND:USW00023183"   # Phoenix
    ]
    
    if !isnothing(stations)
        available_stations = stations[in.(stations.ID, Ref(target_stations)), :]
        println("Stations disponibles:")
        println(available_stations)
    end
    
    # Téléchargement 
    for station in target_stations
        println("\n" * "="^50)
        println("Traitement de la station: $station")
        
        data = download_station_data(station)
        
        if !isnothing(data) && !isempty(data)
            filename = "data/$(replace(station, ":" => "_")).csv"
            CSV.write(filename, data)
            println("✓ Données sauvegardées: $filename ($(nrow(data)) enregistrements)")
            
            # Aperçu des données
            println("Aperçu des données:")
            println(first(data, 5))
            println("Période couverte: $(minimum(data.date)) au $(maximum(data.date))")
        else
            println("✗ Aucune donnée obtenue pour $station")
        end
    end
    
    println("\nTéléchargement terminé!")
end

# Exécuter le programme directement
main()