-- analytics.sql — Enterprise Financial Performance & Fraud Analytics
-- Named queries; the pipeline splits on "-- name:" and runs each block.
-- Dialect: SQLite (portable to Postgres / Snowflake).

-- name: monthly_revenue
-- Company revenue by month (all regions).
SELECT
    month,
    ROUND(SUM(revenue), 0)                     AS revenue
FROM revenue_by_region
GROUP BY month
ORDER BY month;

-- name: revenue_by_region
-- Revenue by region by month (drives the "which region declined" narrative).
SELECT
    region,
    month,
    ROUND(SUM(revenue), 0)                     AS revenue
FROM revenue_by_region
GROUP BY region, month
ORDER BY region, month;

-- name: budget_vs_actual
-- Operating expense: actual vs. budget by department, with variance.
SELECT
    d.name                                     AS department,
    ROUND(SUM(f.opex), 0)                       AS actual_opex,
    ROUND(SUM(f.budget_opex), 0)                AS budget_opex,
    ROUND(SUM(f.opex) - SUM(f.budget_opex), 0)  AS variance,
    ROUND(100.0 * (SUM(f.opex) - SUM(f.budget_opex)) / SUM(f.budget_opex), 1) AS variance_pct
FROM dept_month f
JOIN departments d ON d.dept_id = f.dept_id
GROUP BY d.name
ORDER BY variance DESC;

-- name: ar_aging
-- Accounts-receivable aging buckets on open invoices.
SELECT
    CASE
        WHEN days_outstanding <= 30 THEN '0-30'
        WHEN days_outstanding <= 60 THEN '31-60'
        WHEN days_outstanding <= 90 THEN '61-90'
        ELSE '90+'
    END                                        AS bucket,
    COUNT(*)                                    AS invoices,
    ROUND(SUM(amount), 0)                       AS open_amount
FROM invoices
WHERE status = 'open'
GROUP BY bucket
ORDER BY bucket;
