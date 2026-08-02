"""
pipeline.py  —  Enterprise Financial Performance & Fraud Analytics Platform
===========================================================================
    synthetic data  ->  Python ETL  ->  SQLite  ->  SQL KPIs  ->  ML  ->  exports

  1. generate_data()     Deterministic synthetic financials -> data/*.csv
  2. load_to_sqlite()    ETL into SQLite
  3. run_sql()           Named KPI queries in sql/analytics.sql
  4. financial_kpis()    Revenue, margins, EBITDA, operating cash flow, DSO
  5. detect_fraud()      IsolationForest anomaly detection on transactions
  6. forecast_revenue()  LinearRegression 6-month revenue forecast
  7. exec_summary()      Data-driven "AI" executive summary
  8. export()            metrics.json, dashboard_data.js, dashboard_preview.png

Run:  python src/pipeline.py
Data is illustrative and generated deterministically (numpy seed = 5).
"""

import os, json, sqlite3, datetime as dt, textwrap
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

RNG = np.random.default_rng(5)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SQL  = os.path.join(ROOT, "sql", "analytics.sql")
os.makedirs(DATA, exist_ok=True)

MONTHS = pd.date_range("2024-08-01", periods=12, freq="MS").strftime("%Y-%m").tolist()
REGIONS = ["Northeast", "Midwest", "South", "West"]
DEPTS = ["Sales", "Marketing", "Logistics", "R&D", "Operations", "Finance", "IT", "HR"]
COGS_RATE = 0.55


# ---------------------------------------------------------------- 1. DATA GEN
def generate_data():
    departments = pd.DataFrame({"dept_id": [f"D{i+1}" for i in range(len(DEPTS))],
                                "name": DEPTS})

    # ---- revenue by region x month (Midwest slumps late in the year) ----
    rev_rows = []
    region_base = {"Northeast": 2.1e6, "Midwest": 1.7e6, "South": 2.4e6, "West": 2.9e6}
    for r in REGIONS:
        for i, m in enumerate(MONTHS):
            season = 1 + .06 * np.sin(2 * np.pi * i / 12)
            trend = 1 + .010 * i
            slump = max(.55, 1 - .13 * (i - 8)) if (r == "Midwest" and i >= 9) else 1.0  # progressive decline
            rev = region_base[r] * season * trend * slump * RNG.normal(1, .03)
            rev_rows.append({"region": r, "month": m, "revenue": round(rev, 0)})
    revenue_by_region = pd.DataFrame(rev_rows)

    # ---- dept opex + budget by month (Logistics costs spike) ----
    opex_rows = []
    dept_base = {"Sales": 620000, "Marketing": 410000, "Logistics": 540000, "R&D": 480000,
                 "Operations": 700000, "Finance": 210000, "IT": 330000, "HR": 180000}
    for d in departments.itertuples():
        for i, m in enumerate(MONTHS):
            spike = 1 + (.12 if (d.name == "Logistics" and i >= 9) else 0.0)   # cost overrun
            budget = dept_base[d.name] * (1 + .008 * i)
            actual = budget * spike * RNG.normal(1, .04)
            opex_rows.append({"dept_id": d.dept_id, "month": m,
                              "opex": round(actual, 0), "budget_opex": round(budget, 0)})
    dept_month = pd.DataFrame(opex_rows)

    # ---- transactions (expenses) with injected fraud ----
    vendors = [f"VENDOR-{i:03d}" for i in range(60)]
    cats = ["Travel", "Software", "Consulting", "Supplies", "Facilities", "Equipment"]
    cat_mean = {c: v for c, v in zip(cats, [1800, 4200, 9500, 900, 6200, 12000])}
    tx = []
    n = 4200
    for k in range(n):
        cat = RNG.choice(cats)
        dep = departments.iloc[RNG.integers(0, len(DEPTS))]
        amt = float(max(50, RNG.lognormal(np.log(cat_mean[cat]), .5)))
        hour = int(np.clip(RNG.normal(13, 3), 0, 23))
        date = dt.date(2024, 8, 1) + dt.timedelta(days=int(RNG.integers(0, 360)))
        tx.append({"txn_id": f"T{k+1:05d}", "date": date.isoformat(), "dept_id": dep["dept_id"],
                   "vendor": RNG.choice(vendors), "category": cat, "amount": round(amt, 2),
                   "hour": hour, "is_fraud": 0})
    # inject ~40 anomalies: very large amounts, odd hours, rare vendors, round numbers
    for j in range(40):
        cat = RNG.choice(cats)
        dep = departments.iloc[RNG.integers(0, len(DEPTS))]
        amt = round(cat_mean[cat] * RNG.uniform(6, 14), -3)         # huge, round number
        date = dt.date(2024, 8, 1) + dt.timedelta(days=int(RNG.integers(0, 360)))
        tx.append({"txn_id": f"T9{j:04d}", "date": date.isoformat(), "dept_id": dep["dept_id"],
                   "vendor": f"VENDOR-{RNG.integers(900,999)}", "category": cat,
                   "amount": float(amt), "hour": int(RNG.choice([2, 3, 4, 23])), "is_fraud": 1})
    transactions = pd.DataFrame(tx).sample(frac=1, random_state=5).reset_index(drop=True)

    # ---- invoices / AR ----
    inv = []
    for k in range(1800):
        issue = dt.date(2024, 8, 1) + dt.timedelta(days=int(RNG.integers(0, 360)))
        amount = float(round(RNG.lognormal(np.log(12000), .7), 2))
        due = issue + dt.timedelta(days=30)
        paid = RNG.random() < 0.72
        asof = dt.date(2025, 7, 31)
        if paid:
            paid_date = (issue + dt.timedelta(days=int(RNG.integers(10, 75)))).isoformat()
            status, dso = "paid", 0
        else:
            paid_date, status = "", "open"
            dso = (asof - issue).days
        inv.append({"invoice_id": f"INV{k+1:05d}", "issue_date": issue.isoformat(),
                    "due_date": due.isoformat(), "amount": amount, "paid_date": paid_date,
                    "status": status, "days_outstanding": dso})
    invoices = pd.DataFrame(inv)

    for name, df in [("departments", departments), ("revenue_by_region", revenue_by_region),
                     ("dept_month", dept_month), ("transactions", transactions),
                     ("invoices", invoices)]:
        df.to_csv(os.path.join(DATA, f"{name}.csv"), index=False)
    return dict(departments=departments, revenue_by_region=revenue_by_region,
                dept_month=dept_month, transactions=transactions, invoices=invoices)


# ---------------------------------------------------------------- 2/3. ETL + SQL
def load_to_sqlite(tables):
    conn = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, conn, index=False, if_exists="replace")
    return conn


def parse_named_sql(path):
    blocks, name, buf = {}, None, []
    for line in open(path):
        if line.strip().lower().startswith("-- name:"):
            if name:
                blocks[name] = "".join(buf).strip()
            name, buf = line.split(":", 1)[1].strip(), []
        elif name:
            buf.append(line)
    if name:
        blocks[name] = "".join(buf).strip()
    return blocks


def run_sql(conn):
    return {n: pd.read_sql_query(s.split(";")[0], conn)
            for n, s in parse_named_sql(SQL).items()}


# ---------------------------------------------------------------- 4. KPIs
def financial_kpis(tables, sql_out):
    revenue = tables["revenue_by_region"]["revenue"].sum()
    opex = tables["dept_month"]["opex"].sum()
    cogs = revenue * COGS_RATE
    gross = revenue - cogs
    ebitda = revenue - cogs - opex
    open_ar = tables["invoices"].query("status=='open'")["amount"].sum()
    # simple operating cash flow proxy: EBITDA minus growth in open receivables
    ocf = ebitda - open_ar * 0.35
    dso = round(open_ar / (revenue / 365), 1)
    mr = sql_out["monthly_revenue"]
    rev_mom = round(100 * (mr["revenue"].iloc[-1] - mr["revenue"].iloc[-2]) / mr["revenue"].iloc[-2], 1)
    return {
        "total_revenue": round(revenue, 0),
        "gross_margin_pct": round(100 * gross / revenue, 1),
        "operating_expenses": round(opex, 0),
        "ebitda": round(ebitda, 0),
        "ebitda_margin_pct": round(100 * ebitda / revenue, 1),
        "operating_cash_flow": round(ocf, 0),
        "dso_days": dso,
        "revenue_mom_pct": rev_mom,
    }


# ---------------------------------------------------------------- 5. FRAUD
def detect_fraud(tx):
    df = tx.copy()
    df["is_round"] = (df["amount"] % 1000 == 0).astype(int)
    df["dept_code"] = df["dept_id"].str[1:].astype(int)
    df["vendor_freq"] = df.groupby("vendor")["vendor"].transform("count")
    df["cat_z"] = df.groupby("category")["amount"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    feats = ["amount", "hour", "is_round", "vendor_freq", "dept_code", "cat_z"]
    iso = IsolationForest(n_estimators=250, contamination=0.02, random_state=5).fit(df[feats])
    df["anomaly_score"] = -iso.score_samples(df[feats])     # higher = more anomalous
    df["flagged"] = (iso.predict(df[feats]) == -1).astype(int)

    flagged = df[df["flagged"] == 1]
    # precision against the injected ground-truth anomalies
    precision = round(flagged["is_fraud"].mean(), 3) if len(flagged) else 0.0
    recall = round(df[df["is_fraud"] == 1]["flagged"].mean(), 3)
    top = (flagged.sort_values("anomaly_score", ascending=False)
                  .head(8)[["txn_id", "date", "dept_id", "vendor", "category", "amount", "anomaly_score"]])
    top = top.assign(anomaly_score=top["anomaly_score"].round(3)).to_dict("records")
    return dict(flagged_count=int(df["flagged"].sum()), total=int(len(df)),
                flagged_value=round(float(flagged["amount"].sum()), 0),
                precision_vs_injected=precision, recall_vs_injected=recall, top=top)


# ---------------------------------------------------------------- 6. FORECAST
def forecast_revenue(monthly, horizon=6):
    d = monthly.copy()
    d["idx"] = np.arange(len(d))
    X, y = d[["idx"]].values, d["revenue"].values
    model = LinearRegression().fit(X, y)
    fut = np.arange(len(d), len(d) + horizon).reshape(-1, 1)
    pred = model.predict(fut).round().astype(float)
    fut_months = pd.date_range(pd.to_datetime(d["month"].iloc[-1]) + pd.offsets.MonthBegin(),
                               periods=horizon, freq="MS").strftime("%Y-%m").tolist()
    return dict(history=[{"month": m, "revenue": float(r)} for m, r in zip(d["month"], y)],
                forecast=[{"month": m, "revenue": float(p)} for m, p in zip(fut_months, pred)])


# ---------------------------------------------------------------- 7. SUMMARY
def exec_summary(kpi, region_df, bva, fraud, fc):
    # biggest region MoM revenue decline
    piv = region_df.pivot(index="month", columns="region", values="revenue")
    mom = (piv.iloc[-1] - piv.iloc[-2]) / piv.iloc[-2] * 100
    worst_region = mom.idxmin()
    worst_pct = round(mom.min(), 1)
    top_over = bva.iloc[0]      # largest opex overrun vs budget
    fdir = "decreased" if kpi["revenue_mom_pct"] < 0 else "increased"
    return (
        f"Revenue {fdir} {abs(kpi['revenue_mom_pct'])}% month-over-month to "
        f"${kpi['total_revenue']/1e6:.1f}M, with EBITDA margin at {kpi['ebitda_margin_pct']}% "
        f"and operating cash flow of ${kpi['operating_cash_flow']/1e6:.1f}M. "
        f"The {worst_region} region drove the softness ({worst_pct}% MoM), while "
        f"{top_over['department']} operating expense ran {top_over['variance_pct']}% over budget. "
        f"DSO stands at {kpi['dso_days']} days. Fraud analytics (Isolation Forest) flagged "
        f"{fraud['flagged_count']} anomalous transactions worth ${fraud['flagged_value']/1e6:.2f}M "
        f"for review — {int(fraud['recall_vs_injected']*100)}% of known planted anomalies caught."
    )


# ---------------------------------------------------------------- 8. EXPORT
def export(kpi, sql_out, fraud, fc, summary):
    metrics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "kpi": kpi,
        "monthly_revenue": sql_out["monthly_revenue"].to_dict("records"),
        "revenue_by_region": sql_out["revenue_by_region"].to_dict("records"),
        "budget_vs_actual": sql_out["budget_vs_actual"].to_dict("records"),
        "ar_aging": sql_out["ar_aging"].to_dict("records"),
        "forecast": fc,
        "fraud": fraud,
        "ai_summary": summary,
    }
    json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=2)
    open(os.path.join(ROOT, "dashboard_data.js"), "w").write(
        "window.FIN_DATA = " + json.dumps(metrics) + ";")
    render_preview(metrics)
    return metrics


def render_preview(m):
    BG, PANEL, INK, MUT, LINE = "#0b1120", "#18213a", "#e8eefc", "#94a3c6", "#27324f"
    ACC = ["#34d399", "#38bdf8", "#f472b6", "#fbbf24"]
    fig = plt.figure(figsize=(13, 8.4), facecolor=BG)
    fig.suptitle("Enterprise Financial Performance & Fraud Analytics",
                 x=.065, ha="left", color=INK, fontsize=19, fontweight="bold", y=.975)
    fig.text(.065, .935, "Revenue · margins · budget vs actual · fraud detection · synthetic data",
             color=MUT, fontsize=11)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[.85, 2, 1.15],
                           hspace=.55, wspace=.32, left=.06, right=.965, top=.9, bottom=.055)

    def panel(ax): ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]
    esc = lambda s: str(s).replace("$", r"\$")   # matplotlib reads $ as math mode
    k = m["kpi"]
    cards = [("REVENUE", f"${k['total_revenue']/1e6:.1f}M", ACC[0]),
             ("EBITDA MARGIN", f"{k['ebitda_margin_pct']}%", ACC[1]),
             ("OPERATING CASH FLOW", f"${k['operating_cash_flow']/1e6:.1f}M", ACC[3]),
             ("FLAGGED FOR FRAUD", f"{m['fraud']['flagged_count']}", ACC[2])]
    for i, (lbl, val, c) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); panel(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.text(.08, .66, lbl, transform=ax.transAxes, color=MUT, fontsize=8.5)
        ax.text(.08, .26, esc(val), transform=ax.transAxes, color=c, fontsize=22, fontweight="bold")

    # revenue trend + forecast
    axr = fig.add_subplot(gs[1, 0:2]); panel(axr)
    h = m["forecast"]["history"]; f = m["forecast"]["forecast"]
    axr.plot(range(len(h)), [x["revenue"]/1e6 for x in h], color=ACC[1], lw=2, label="Actual")
    axr.plot(range(len(h)-1, len(h)+len(f)),
             [h[-1]["revenue"]/1e6] + [x["revenue"]/1e6 for x in f], color=ACC[2], lw=2, ls="--", label="Forecast")
    axr.set_title("Monthly revenue — 6-month forecast ($M)", color=INK, fontsize=12, loc="left", pad=8)
    axr.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=INK, fontsize=9)
    axr.tick_params(colors=MUT, labelsize=8); axr.grid(color=LINE, ls=":", alpha=.6)
    for s in axr.spines.values(): s.set_color(LINE)

    # budget vs actual variance
    axb = fig.add_subplot(gs[1, 2:4]); panel(axb)
    bva = m["budget_vs_actual"][:8][::-1]
    names = [b["department"] for b in bva]
    varpct = [b["variance_pct"] for b in bva]
    axb.barh(names, varpct, color=[ACC[2] if v > 3 else ACC[0] if v < -3 else ACC[3] for v in varpct])
    axb.axvline(0, color=MUT, lw=.8)
    axb.set_title("Opex variance vs budget (%)", color=INK, fontsize=12, loc="left", pad=8)
    axb.tick_params(colors=MUT, labelsize=9); axb.grid(axis="x", color=LINE, ls=":", alpha=.6)
    for s in axb.spines.values(): s.set_color(LINE)

    axs = fig.add_subplot(gs[2, :]); panel(axs); axs.set_xticks([]); axs.set_yticks([])
    axs.text(.015, .82, "AI EXECUTIVE SUMMARY", transform=axs.transAxes,
             color=ACC[1], fontsize=12, fontweight="bold")
    axs.text(.015, .60, esc("\n".join(textwrap.wrap(m["ai_summary"], 118))),
             transform=axs.transAxes, color=MUT, fontsize=10.3, va="top")
    fig.savefig(os.path.join(ROOT, "dashboard_preview.png"), dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    t = generate_data()
    conn = load_to_sqlite(t)
    sql_out = run_sql(conn)
    kpi = financial_kpis(t, sql_out)
    fraud = detect_fraud(t["transactions"])
    fc = forecast_revenue(sql_out["monthly_revenue"])
    summary = exec_summary(kpi, sql_out["revenue_by_region"], sql_out["budget_vs_actual"], fraud, fc)
    export(kpi, sql_out, fraud, fc, summary)

    print("=== Enterprise Financial Performance & Fraud Analytics — complete ===")
    print(f"Total revenue      : ${kpi['total_revenue']/1e6:.1f}M   EBITDA margin: {kpi['ebitda_margin_pct']}%")
    print(f"Operating cash flow: ${kpi['operating_cash_flow']/1e6:.1f}M   DSO: {kpi['dso_days']}d   Rev MoM: {kpi['revenue_mom_pct']:+.1f}%")
    print(f"Fraud flagged      : {fraud['flagged_count']}/{fraud['total']}  "
          f"(recall on planted {int(fraud['recall_vs_injected']*100)}%, precision {fraud['precision_vs_injected']})")
    print("Exports: metrics.json, dashboard_data.js, dashboard_preview.png")


if __name__ == "__main__":
    main()
