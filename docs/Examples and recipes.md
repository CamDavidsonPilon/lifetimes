## Examples and recipes

### Example SQL statement to transform transactional data into RFM data

Let's review what our variables mean: 

- `frequency` represents the number of *repeat* purchases the customer has made. This means that it's one less than the total number of purchases. This is actually slightly wrong. It's the count of distinct time periods the customer had a purchase in. So if using days as units, then it's the count of distinct days the customer had a purchase on.   
- `T` represents the age of the customer in whatever time units chosen. This is equal to the duration between a customer's first purchase and the end of the period under study.
- `recency` represents the age of the customer when they made their most recent purchases. This is equal to the duration between a customer's first purchase and their latest purchase. (Thus if they have made only 1 purchase, the recency is 0.)

Thus, executing a query against a transactional dataset, called `orders`, in a SQL-store may look like:

.. code-block:: mysql


    SELECT
      customer_id,
      COUNT(distinct date(transaction_at)) - 1 as frequency,
      datediff('day', MIN(transaction_at), MAX(transaction_at)) as recency,
      datediff('day', CURRENT_DATE, MIN(transaction_at)) as T
    FROM orders
    GROUP BY customer_id