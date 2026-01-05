

select
    cast("VISIT KEY" as integer)      as visit_key,
    cast("RESORT KEY" as integer)     as resort_key,
    cast("CUSTOMER KEY" as integer)   as customer_key,
    "VISIT START DATE"::timestamp      as visit_start_date,
    "VISIT END DATE"::timestamp        as visit_end_date
from read_csv_auto(
    '../Data/Bronze/guest_visitation.txt',
    header=true
)