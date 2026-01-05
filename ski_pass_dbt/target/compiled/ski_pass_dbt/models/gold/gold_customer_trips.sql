

with base as (
    select 
        gd.customer_key,
        gd.age,
        gd.renewal_in_subsequent_season,
        gd.guest_state,
        gv.visit_key,
        gv.resort_key,
        gv.visit_start_date,
        gv.visit_end_date,
        gt.purchase_key,
        upper(gt.purchase_product) as purchase_product,
        rd.resort_state
    from "ski"."main"."silver_guest_demographics" gd
    inner join "ski"."main"."silver_guest_visitation" gv
        on gd.customer_key = gv.customer_key
    inner join "ski"."main"."silver_resort_dimensions" rd
        on gv.resort_key = rd.resort_key
    left join "ski"."main"."silver_guest_transactions" gt
        on gv.customer_key = gt.customer_key
       and gv.visit_key = gt.visit_key
),

-- Feature flags
flags as (
    select
        *,
        case when guest_state != resort_state then 1 else 0 end as fe_out_of_state_trip,
        date_diff('day', visit_start_date, visit_end_date) + 1 as fe_trip_length_in_days,
        case when extract(dow from visit_start_date) in (0,6) then 1 else 0 end as fe_is_weekend
    from base
),

-- Unique trips per customer
unique_trips as (
    select distinct customer_key, visit_key, fe_out_of_state_trip, fe_trip_length_in_days, fe_is_weekend,
           visit_start_date, visit_end_date
    from flags
),

-- Compute gaps first
trips_with_gaps as (
    select
        *,
        date_diff('day', lag(visit_end_date) over (partition by customer_key order by visit_start_date), visit_start_date) as gap_days
    from unique_trips
),

-- Aggregated features per customer
customer_agg as (
    select
        customer_key,
        min(fe_trip_length_in_days) as min_trip_length,
        max(fe_trip_length_in_days) as max_trip_length,
        avg(fe_trip_length_in_days) as avg_trip_length,
        sum(fe_trip_length_in_days) as total_days_on_mountain,
        sum(case when fe_out_of_state_trip = 0 then fe_trip_length_in_days else 0 end) as total_days_in_state,
        sum(case when fe_out_of_state_trip = 1 then fe_trip_length_in_days else 0 end) as total_days_out_of_state,
        count(visit_key) as total_trips,
        sum(case when fe_out_of_state_trip = 0 then 1 else 0 end) as total_trips_in_state,
        sum(case when fe_out_of_state_trip = 1 then 1 else 0 end) as total_trips_out_of_state,
        avg(fe_out_of_state_trip) as pct_trips_out_of_state,
        max(gap_days) as biggest_gap_days,
        min(gap_days) as smallest_gap_days,
        avg(gap_days) as avg_gap_days
    from trips_with_gaps
    group by customer_key
)

select * from customer_agg