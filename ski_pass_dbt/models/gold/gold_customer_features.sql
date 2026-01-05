{{ config(materialized='view') }}

with trips as (

    select *
    from {{ ref('gold_customer_trips') }}

),

purchases as (

    select *
    from {{ ref('gold_customer_purchases') }}

),

demographics as (

    select
        customer_key,
        age,
        guest_state,
        renewal_in_subsequent_season
    from {{ ref('silver_guest_demographics') }}

)

select
    -- -------------------------------------------------
    -- Identifiers
    -- -------------------------------------------------
    t.customer_key,

    -- -------------------------------------------------
    -- Label & static customer attributes
    -- -------------------------------------------------
    d.renewal_in_subsequent_season,
    d.age,
    d.guest_state,

    -- -------------------------------------------------
    -- Trip behavior (raw)
    -- -------------------------------------------------
    t.min_trip_length,
    t.max_trip_length,
    t.avg_trip_length,
    t.total_days_on_mountain,
    t.total_days_in_state,
    t.total_days_out_of_state,
    t.total_trips,
    t.total_trips_in_state,
    t.total_trips_out_of_state,
    t.pct_trips_out_of_state,
    t.biggest_gap_days,
    t.smallest_gap_days,
    t.avg_gap_days,

    -- -------------------------------------------------
    -- Purchase behavior (raw)
    -- -------------------------------------------------
    coalesce(p.total_rentals, 0) as total_rentals,
    coalesce(p.total_lessons, 0) as total_lessons,

    -- -------------------------------------------------
    -- ML-derived features (previously in Python)
    -- -------------------------------------------------

    -- Single-trip customer flag
    case
        when t.total_trips = 1 then 1
        else 0
    end as is_single_trip_customer,

    -- Zero-inflation flags
    case
        when t.total_trips_out_of_state > 0 then 1
        else 0
    end as total_trips_out_of_state_nonzero,

    case
        when t.total_days_out_of_state > 0 then 1
        else 0
    end as total_days_out_of_state_nonzero,

    case
        when t.pct_trips_out_of_state > 0 then 1
        else 0
    end as pct_trips_out_of_state_nonzero,

    case
        when coalesce(p.total_lessons, 0) > 0 then 1
        else 0
    end as total_lessons_nonzero,

    -- Gap null handling (explicit, stable)
    coalesce(t.biggest_gap_days, 0)  as biggest_gap_days_filled,
    coalesce(t.smallest_gap_days, 0) as smallest_gap_days_filled,
    coalesce(t.avg_gap_days, 0)      as avg_gap_days_filled

from trips t
left join purchases p
    on t.customer_key = p.customer_key
left join demographics d
    on t.customer_key = d.customer_key
