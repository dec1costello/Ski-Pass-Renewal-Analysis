
  
  create view "ski"."main"."gold_customer_purchases__dbt_tmp" as (
    

with purchases as (
    select 
        customer_key,
        sum(case when lower(purchase_product) = 'rental' then 1 else 0 end) as total_rentals,
        sum(case when lower(purchase_product) = 'lesson' then 1 else 0 end) as total_lessons
    from "ski"."main"."silver_guest_transactions"
    group by customer_key
)

select * from purchases
  );
