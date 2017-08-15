## More Examples and recipes

### Example SQL statement to transform transactional data into RFM data

Let's review what our variables mean:

- `frequency` represents the number of *repeat* purchases the customer has made. This means that it's one less than the total number of purchases. This is actually slightly wrong. It's the count of distinct time periods the customer had a purchase in. So if using days as units, then it's the count of distinct days the customer had a purchase on.   
- `T` represents the age of the customer in whatever time units chosen. This is equal to the duration between a customer's first purchase and the end of the period under study.
- `recency` represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

Thus, executing a query against a transactional dataset, called `orders`, in a SQL-store may look like:

```sql
SELECT
  customer_id,
  COUNT(distinct date(transaction_at)) - 1 as frequency,
  datediff('day', MIN(transaction_at), MAX(transaction_at)) as recency,
  datediff('day', CURRENT_DATE, MIN(transaction_at)) as T
FROM orders
GROUP BY customer_id
```

### Create table with RFM summary matrix with holdout

Variables `frequency`, `T` and `recency` have the same meaning as in previous section.

Two variables to set before executing:

- `duration_holdout` - holdout duration in days.
- `CURRENT_DATE` - current date, could be changed to final date of the transactional data.

```sql
select
    a.*,
    COALESCE(b.frequency_holdout, 0) as frequency_holdout,
    duration_holdout as duration_holdout
from (
    select
        customer_id,
        datediff(max(event_date), min(event_date)) as recency,
        count(*) - 1 as frequency,
        datediff(date_sub(CURRENT_DATE, duration_holdout), min(event_date)) as T
    from orders
    where event_date < date_sub(CURRENT_DATE, duration_holdout)
    group by customer_id
) a
left join (
    select
        customer_id,
        count(*) as frequency_holdout
    from orders
    where event_date >= date_sub(CURRENT_DATE, duration_holdout)
      and event_date < CURRENT_DATE
    group by customer_id
) b
on a.customer_id = b.customer_id
```
