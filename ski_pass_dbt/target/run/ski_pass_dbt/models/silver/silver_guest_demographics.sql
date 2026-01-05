
  
    
    

    create  table
      "ski"."main"."silver_guest_demographics__dbt_tmp"
  
    as (
      

select
    cast("CUSTOMER KEY" as integer)    as customer_key,
    cast("AGE" as integer)             as age,
    cast("RENEWALINSUBSEQUENTSEASON" as boolean) as renewal_in_subsequent_season,
    cast("STATE" as varchar)           as guest_state
from read_csv_auto(
    '../Data/Bronze/guest_demographics.txt',
    header=true
)
    );
  
  