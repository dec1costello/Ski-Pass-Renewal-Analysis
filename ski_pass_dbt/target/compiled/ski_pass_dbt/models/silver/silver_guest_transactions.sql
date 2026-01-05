

select
    cast("PURCHASE KEY" as integer)   as purchase_key,
    cast("VISIT KEY" as integer)      as visit_key,
    cast("CUSTOMER KEY" as integer)   as customer_key,
    "PURCHASE DATE"::timestamp         as purchase_date,
    upper("PURCHASE PRODUCT")          as purchase_product
from read_csv_auto(
    '../Data/Bronze/guest_transactions.txt',
    header=true
)