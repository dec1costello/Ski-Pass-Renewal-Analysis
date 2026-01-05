{{ config(materialized='table') }}

select
    cast("RESORT KEY" as integer) as resort_key,
    upper("RESORT")               as resort,
    cast("STATE" as varchar)      as resort_state
from read_csv_auto(
    '../Data/Bronze/resort_dimensions.txt',
    header=true
)
