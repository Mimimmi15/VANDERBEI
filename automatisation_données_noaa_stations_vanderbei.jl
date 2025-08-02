
using Pkg
#Pkg.add("Revise")
Pkg.add(["HTTP", "CSV", "DataFrames", "Dates", "JSON3","PyCall"])
#using Revise
using PyCall
pyimport("conda").activate("spyder-runtime")
#Revise.clear() 
using HTTP
using CSV
using DataFrames
using Dates: Date
using JSON3

# Nettoyage préventif
if @isdefined(station_id)
    @warn "Nettoyage variable globale station_id"
    station_id = nothing
end


const NOAA_API_KEY = get(ENV, "NOAA_API_KEY","cBJbvohaShpTOUXRxRiwDxFqybZWCyoA")
const BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
const HEADERS = Dict("token" => NOAA_API_KEY)

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
        @error "Échec du téléchargement de la liste des stations" exception=(e, catch_backtrace())
        return nothing
    end
end

function download_station_data(station_id; start_date="1955-01-01", end_date="2010-12-31", datatype="TAVG")
    try
        start_dt = Date(start_date)
        end_dt = Date(end_date)
    catch e
        @error "Format de date invalide" start_date end_date exception=e
        return nothing
    end

    all_data = DataFrame()
    year_ranges = start_dt:Year(1):end_dt
    
    for (i, year_start) in enumerate(year_ranges)
        year_end = i < length(year_ranges) ? year_ranges[i+1] - Day(1) : end_dt
        
        params = Dict(
            "datasetid" => "GHCND",
            "stationid" => station_id,
            "startdate" => string(year_start),
            "enddate" => string(year_end),
            "datatypeid" => datatype,
            "units" => "standard",
            "limit" => 1000
        )
        
        @info "Téléchargement en cours" station=station_id période="$year_start - $year_end"
        
        try
            response = HTTP.get("$BASE_URL/data", headers=HEADERS, query=params)
            
            if response.status != 200
                @warn "Erreur API" status=response.status station=station_id
                continue
            end
            
            json_data = JSON3.read(response.body)
            if !haskey(json_data, :results) || isempty(json_data.results)
                @info "Aucune donnée disponible" station=station_id période="$year_start - $year_end"
                continue
            end
            
            chunk = DataFrame(json_data.results)
            all_data = vcat(all_data, chunk, cols=:union)
            sleep(1.2)  # Marge de sécurité
            
        catch e
            @error "Erreur de traitement" station=station_id exception=(e, catch_backtrace())
        end
    end
    
    if !isempty(all_data)
        transform!(all_data, 
            :date => ByRow(d -> Date(d[1:10])) => :date,
            :value => ByRow(v -> v/10) => :temp_f
        )
    end
    
    return all_data
end

function main()
    mkpath("data")
    target_stations = ["GHCND:USW00013743", "GHCND:USW00094728", "GHCND:USW00023183"]
    
    for station in target_stations
        @info "Traitement de la station" station=station
        data = download_station_data(station)
        
        if !isnothing(data) && !isempty(data)
            filename = "data/$(replace(station, ":" => "_")).csv"
            CSV.write(filename, data)
            @info "Données sauvegardées" fichier=filename nblignes=nrow(data)
        else
            @warn "Aucune donnée obtenue" station=station
        end
    end
end

main()



