# app.py
import streamlit as st
import psycopg2
from psycopg2 import OperationalError
from psycopg2.extras import execute_values
import pandas as pd
import datetime

# ==============================
# Page setup & modern styling
# ==============================
st.set_page_config(page_title="Manufacturing Data Entry", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{ --bg1:#0b0f19; --bg2:#0f172a; --panel:rgba(255,255,255,.04);
--border:rgba(255,255,255,.08); --text:#e5e7eb; --muted:#94a3b8;
--accent:#22d3ee; --accent2:#10b981; }
html, body, .block-container{ background: radial-gradient(1200px 600px at 10% 0%, var(--bg2), var(--bg1)) !important; }
.block-container{ padding-top: 1.2rem; padding-bottom: .75rem; }
.hero{ display:flex; gap:14px; align-items:center; padding:14px 16px; border-radius:16px;
  border:1px solid var(--border);
  background: linear-gradient(90deg, rgba(34,211,238,.14), rgba(16,185,129,.12));
  box-shadow: 0 6px 22px rgba(0,0,0,.25); backdrop-filter: blur(8px); margin: 0 0 12px 0;}
.hero .emoji{ font-size: 26px; } .hero h1{ margin:0; line-height:1.15; font-size: 1.8rem;
  background: linear-gradient(90deg,#ffffff,#a7f3d0); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero .sub{ margin:2px 0 0 0; color: var(--muted); font-size:.92rem; }
div[data-testid="stForm"]{ background:var(--panel); border:1px solid var(--border);
  border-radius:16px; padding:12px !important; box-shadow: 0 4px 14px rgba(0,0,0,.18);}
.stSelectbox, .stNumberInput, .stDateInput, .stTextInput{ margin-bottom:.35rem; }
.stButton>button{ width:100%; border-radius:12px; padding:.62rem .9rem; font-weight:600;
  border:1px solid rgba(34,211,238,.25);
  background: linear-gradient(180deg, rgba(34,211,238,.18), rgba(34,211,238,.12));}
.card{ border-radius:16px; padding:12px 14px; background:var(--panel);
  border:1px solid var(--border); box-shadow: 0 4px 14px rgba(0,0,0,.18);}
.small-label{ font-size:.82rem; color:var(--muted); margin-bottom:.25rem; }
.metric-value{ font-size:1.6rem; font-weight:800; color:var(--text); }
div[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:16px; background:var(--panel); }
</style>
""", unsafe_allow_html=True)

# ==============================
# Database helpers (from st.secrets)
# ==============================
def _load_pg_secrets():
    try:
        return dict(st.secrets["pg"])
    except Exception:
        st.error("Database credentials not found in st.secrets['pg'].\n"
                 "Create `.streamlit/secrets.toml` with:\n\n"
                 "[pg]\nhost='127.0.0.1'\ndatabase='manufacturing_dw'\nuser='postgres'\npassword='YOUR_PASSWORD'\nport=5432")
        st.stop()

@st.cache_resource
def get_connection():
    pg = _load_pg_secrets()
    try:
        conn = psycopg2.connect(
            host=pg.get("host", "127.0.0.1"),
            database=pg["database"],
            user=pg["user"],
            password=pg["password"],
            port=int(pg.get("port", 5432)),
        )
        conn.autocommit = True
        return conn
    except OperationalError as e:
        st.error("❌ Database connection failed. Check host/port/user/password in `.streamlit/secrets.toml`.\n"
                 "Tip: use `host='127.0.0.1'` instead of `localhost`.")
        st.exception(e)
        st.stop()

@st.cache_data(ttl=600)
def fetch_options(q: str) -> pd.DataFrame:
    return pd.read_sql(q, get_connection())

# DIMs
machines_df  = fetch_options("SELECT machinekey, machinename FROM dimmachine;")
shifts_df    = fetch_options("SELECT shiftkey, shiftname FROM dimshift;")
operators_df = fetch_options("SELECT operatorkey, operatorname FROM dimoperator;")
products_df  = fetch_options("SELECT productkey, productname FROM dimproduct;")

# Normalized (case/space-insensitive) name→key maps
def _norm(x):
    return str(x).strip().casefold() if pd.notna(x) else None

shift_map    = {_norm(n): k for n, k in zip(shifts_df["shiftname"],    shifts_df["shiftkey"])}
machine_map  = {_norm(n): k for n, k in zip(machines_df["machinename"], machines_df["machinekey"])}
operator_map = {_norm(n): k for n, k in zip(operators_df["operatorname"], operators_df["operatorkey"])}
product_map  = {_norm(n): k for n, k in zip(products_df["productname"],  products_df["productkey"])}

# ==============================
# Header
# ==============================
st.markdown(
    '<div class="hero"><div class="emoji">🏭</div>'
    '<div><h1>Manufacturing Data Entry</h1>'
    '<div class="sub">Record production in seconds • Live OEE on the right • Admin tools below</div></div></div>',
    unsafe_allow_html=True
)

# ==============================
# Recent records loader (names + refresh)
# ==============================
@st.cache_data(ttl=300)
def load_recent_records(limit: int = 10, refresh_token: int = 0) -> pd.DataFrame:
    """
    Return latest records with DIM names instead of keys.
    refresh_token busts the cache when you click Refresh.
    """
    sql = """
    SELECT
        f.datekey,
        TO_CHAR(TO_DATE(f.datekey::text,'YYYYMMDD'), 'YYYY-MM-DD') AS date,
        s.shiftname,
        m.machinename,
        o.operatorname,
        p.productname,
        f.scheduledtime_min,
        f.downtime_min,
        f.runtime_min,
        f.actualproduction,
        f.rejectcount,
        f.availability_pct,
        f.performance_pct,
        f.quality_pct,
        f.oee_pct
    FROM factproduction f
    LEFT JOIN dimshift    s ON s.shiftkey    = f.shiftkey
    LEFT JOIN dimmachine  m ON m.machinekey  = f.machinekey
    LEFT JOIN dimoperator o ON o.operatorkey = f.operatorkey
    LEFT JOIN dimproduct  p ON p.productkey  = f.productkey
    ORDER BY f.datekey DESC, f.productkey, f.machinekey
    LIMIT %(limit)s;
    """
    df = pd.read_sql(sql, get_connection(), params={"limit": limit})
    cols = [
        "date", "shiftname", "machinename", "operatorname", "productname",
        "scheduledtime_min", "downtime_min", "runtime_min",
        "actualproduction", "rejectcount",
        "availability_pct", "performance_pct", "quality_pct", "oee_pct",
        "datekey"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df

# ==============================
# Main layout
# ==============================
left, right = st.columns([1.5, 1])

# ---------- Single-Row Entry Form ----------
with left:
    with st.form("production_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            date_sel = st.date_input(
                "Date", value=datetime.date.today(),
                min_value=datetime.date(2025, 1, 1), max_value=datetime.date.today()
            )
        with c2:
            shift_sel = st.selectbox("Shift", shifts_df["shiftname"].tolist())
        with c3:
            machine_sel = st.selectbox("Machine", machines_df["machinename"].tolist())

        c4, c5, c6 = st.columns(3)
        with c4:
            operator_sel = st.selectbox("Operator", operators_df["operatorname"].tolist())
        with c5:
            product_sel  = st.selectbox("Product",  products_df["productname"].tolist())
        with c6:
            scheduled_time = st.number_input("Scheduled Time (min)", min_value=0, value=480, step=10)

        c7, c8, c9 = st.columns(3)
        with c7:
            downtime = st.number_input("Downtime (min)", min_value=0, value=0, step=5)
        runtime = max(scheduled_time - downtime, 0)
        with c8:
            st.markdown('<div class="small-label">Runtime (min)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="metric-value">{runtime}</div></div>', unsafe_allow_html=True)
        with c9:
            production = st.number_input("Actual Production", min_value=0, value=0, step=1)

        c10, c11, c12 = st.columns(3)
        with c10:
            rejects = st.number_input("Reject Count", min_value=0, value=0, step=1)

        # Derived KPIs (live)
        availability = (runtime / scheduled_time) * 100 if scheduled_time > 0 else 0
        ideal_cycle  = 1.0  # replace with product/machine-specific cycle if available
        performance  = (production * ideal_cycle / runtime) * 100 if runtime > 0 else 0
        quality      = ((production - rejects) / production) * 100 if production > 0 else 0
        oee          = (availability * performance * quality) / 10000

        with c11:
            st.markdown('<div class="small-label">OEE (%)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="metric-value">{oee:.2f}</div></div>', unsafe_allow_html=True)
        with c12:
            submitted = st.form_submit_button("Submit Production Record")

        if submitted:
            conn = get_connection()
            cur = conn.cursor()
            datekey = int(date_sel.strftime("%Y%m%d"))

            # Upsert dimdate
            cur.execute("""
                INSERT INTO dimdate(datekey, date, year, month, day, weekday)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT(datekey) DO NOTHING;
            """, (datekey, date_sel, date_sel.year, date_sel.month, date_sel.day, date_sel.strftime("%A")))

            # Key lookups
            shiftkey   = int(shifts_df.loc[shifts_df["shiftname"] == shift_sel, "shiftkey"].iloc[0])
            machinekey = int(machines_df.loc[machines_df["machinename"] == machine_sel, "machinekey"].iloc[0])
            operkey    = int(operators_df.loc[operators_df["operatorname"] == operator_sel, "operatorkey"].iloc[0])
            prodkey    = int(products_df.loc[products_df["productname"] == product_sel, "productkey"].iloc[0])

            cur.execute("""
                INSERT INTO factproduction
                (datekey, shiftkey, machinekey, operatorkey, productkey,
                 scheduledtime_min, downtime_min, runtime_min,
                 actualproduction, rejectcount,
                 availability_pct, performance_pct, quality_pct, oee_pct)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (datekey, shiftkey, machinekey, operkey, prodkey,
                  scheduled_time, downtime, runtime,
                  production, rejects,
                  round(availability,2), round(performance,2),
                  round(quality,2), round(oee,2)))
            conn.commit(); cur.close()
            st.success("✅ Production record inserted!")
            # Optional toast
            st.toast("Saved!", icon="✅")

    # ---------- Recent records (names + refresh) ----------
    with st.expander("Recent records (last 10)"):
        if "recent_refresh" not in st.session_state:
            st.session_state["recent_refresh"] = 0

        c1, c2, c3 = st.columns([0.18, 0.22, 0.60])
        with c1:
            if st.button("↻ Refresh", use_container_width=True):
                st.session_state["recent_refresh"] += 1
        with c2:
            limit = st.number_input("Rows", min_value=5, max_value=100, value=10, step=5, label_visibility="visible")
        with c3:
            st.caption("Showing names (shift/machine/operator/product) for readability.")

        recent_df = load_recent_records(limit=int(limit), refresh_token=st.session_state["recent_refresh"])
        st.dataframe(recent_df, use_container_width=True, height=280)

    # ==============================
    # Admin: Import CSV / Delete Data
    # ==============================
    with st.expander("🛠️ Admin: Import CSV / Delete Data", expanded=False):
        t1, t2 = st.tabs(["📥 Import CSV", "🧹 Delete Data"])

        # ---------- Import CSV (Names template only) ----------
        with t1:
            st.caption("Upload a CSV using **names** (recommended). "
                       "We map names → keys and compute runtime/metrics if missing.")

            # Template (names-based)
            st.markdown("##### Template (names-based)")
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            names_sample = pd.DataFrame([{
                "date": today_str,  # YYYY-MM-DD (ISO)
                "shiftname":   (shifts_df["shiftname"].iat[0]      if not shifts_df.empty      else "Shift_A"),
                "machinename": (machines_df["machinename"].iat[0]  if not machines_df.empty   else "Machine_01"),
                "operatorname":(operators_df["operatorname"].iat[0] if not operators_df.empty else "Operator_01"),
                "productname": (products_df["productname"].iat[0]  if not products_df.empty   else "Product_A"),
                "scheduledtime_min": 480,
                "downtime_min": 0,
                "actualproduction": 0,
                "rejectcount": 0
            }])
            names_blank = names_sample.iloc[0:0]
            st.dataframe(names_sample, use_container_width=True, height=140)
            st.download_button(
                label="⬇️ Download names template (CSV)",
                data=names_blank.to_csv(index=False).encode("utf-8"),
                file_name="factproduction_template_names.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.divider()

            file = st.file_uploader("CSV file to import", type=["csv"])
            dry_run = st.checkbox("Dry run (validate only)", value=False)

            def safe_div(num, den):
                s = pd.Series(num, dtype="float64")
                d = pd.Series(den, dtype="float64")
                out = s.divide(d.where((d != 0) & d.notna(), pd.NA))
                return out.fillna(0.0)

            if file:
                try:
                    df = pd.read_csv(file)
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
                    df = None

                if df is not None:
                    st.write("Preview:", df.head())
                    st.info(f"Rows: {len(df):,} • Columns: {list(df.columns)}")

                    # ---- Enrichment / normalization ----
                    # Map names → keys (case/space-insensitive)
                    if "shiftkey" not in df.columns and "shiftname" in df.columns:
                        df["shiftkey"] = df["shiftname"].map(lambda v: shift_map.get(_norm(v)))
                    if "machinekey" not in df.columns and "machinename" in df.columns:
                        df["machinekey"] = df["machinename"].map(lambda v: machine_map.get(_norm(v)))
                    if "operatorkey" not in df.columns and "operatorname" in df.columns:
                        df["operatorkey"] = df["operatorname"].map(lambda v: operator_map.get(_norm(v)))
                    if "productkey" not in df.columns and "productname" in df.columns:
                        df["productkey"] = df["productname"].map(lambda v: product_map.get(_norm(v)))

                    # datekey from date if missing (accepts ISO or day-first like 11/8/2025)
                    if "datekey" not in df.columns and "date" in df.columns:
                        parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
                        df["datekey"] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce")

                    # runtime if missing
                    if "runtime_min" not in df.columns and {"scheduledtime_min","downtime_min"}.issubset(df.columns):
                        df["runtime_min"] = (pd.to_numeric(df["scheduledtime_min"], errors="coerce") -
                                             pd.to_numeric(df["downtime_min"], errors="coerce")).clip(lower=0)

                    # Compute metrics if missing
                    metric_cols = ["availability_pct","performance_pct","quality_pct","oee_pct"]
                    need_metrics = any(c not in df.columns for c in metric_cols)
                    if need_metrics:
                        st.caption("Computing missing metrics: availability, performance, quality, OEE.")
                        sched = pd.to_numeric(df.get("scheduledtime_min", 0), errors="coerce").fillna(0)
                        run   = pd.to_numeric(df.get("runtime_min", 0), errors="coerce").fillna(0)
                        prod  = pd.to_numeric(df.get("actualproduction", 0), errors="coerce").fillna(0)
                        rej   = pd.to_numeric(df.get("rejectcount", 0), errors="coerce").fillna(0)

                        df["availability_pct"] = (100 * safe_div(run, sched)).round(2)
                        df["performance_pct"]  = (100 * safe_div(prod, run)).round(2)
                        q = (prod - rej)
                        df["quality_pct"]      = (100 * safe_div(q, prod.where(prod>0, pd.NA))).round(2)
                        df["oee_pct"]          = ((df["availability_pct"] * df["performance_pct"] * df["quality_pct"]) / 10000).round(2)

                    # Validate required columns
                    base_cols = ["datekey","shiftkey","machinekey","operatorkey","productkey",
                                 "scheduledtime_min","downtime_min","runtime_min",
                                 "actualproduction","rejectcount"]
                    required = base_cols + metric_cols

                    # Coerce numeric & drop invalid rows
                    for c in base_cols:
                        df[c] = pd.to_numeric(df.get(c), errors="coerce")
                    for c in metric_cols:
                        df[c] = pd.to_numeric(df.get(c), errors="coerce")

                    # Warn on unmapped names
                    issues = []
                    for col_name, map_dict in [("shiftname", shift_map),
                                               ("machinename", machine_map),
                                               ("operatorname", operator_map),
                                               ("productname", product_map)]:
                        if col_name in df.columns:
                            key_col = col_name.replace("name", "key")
                            if key_col in df.columns:
                                unmapped = df.loc[df[key_col].isna(), col_name].dropna().unique().tolist()
                                if unmapped:
                                    issues.append(f"{col_name}: {unmapped[:10]}{'...' if len(unmapped)>10 else ''}")
                    if issues:
                        st.warning("Some names could not be mapped to keys:\n- " + "\n- ".join(issues))

                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        st.error(f"Missing required columns after processing: {missing}")
                    else:
                        before = len(df)
                        df = df.dropna(subset=["datekey","shiftkey","machinekey","operatorkey","productkey"]).copy()
                        after = len(df)
                        if after < before:
                            st.warning(f"Dropped {before-after:,} row(s) missing mandatory key fields.")

                        # Round & cast
                        for c in metric_cols:
                            df[c] = df[c].fillna(0).round(2)
                        int_cols = ["datekey","shiftkey","machinekey","operatorkey","productkey",
                                    "scheduledtime_min","downtime_min","runtime_min",
                                    "actualproduction","rejectcount"]
                        for c in int_cols:
                            df[c] = df[c].fillna(0).astype(int)

                        if st.button("Validate & (optionally) Import"):
                            conn = get_connection()
                            cur = conn.cursor()
                            try:
                                # Upsert dimdate rows (unique)
                                dmap = pd.DataFrame({"datekey": df["datekey"].unique()})
                                dmap["date"] = pd.to_datetime(dmap["datekey"].astype(str), format="%Y%m%d", errors="coerce")
                                dmap = dmap.dropna().astype({"datekey":"int64"})
                                dmap["year"] = dmap["date"].dt.year
                                dmap["month"] = dmap["date"].dt.month
                                dmap["day"] = dmap["date"].dt.day
                                dmap["weekday"] = dmap["date"].dt.day_name()

                                if len(dmap):
                                    execute_values(
                                        cur,
                                        """
                                        INSERT INTO dimdate(datekey, date, year, month, day, weekday)
                                        VALUES %s
                                        ON CONFLICT(datekey) DO NOTHING
                                        """,
                                        list(dmap[["datekey","date","year","month","day","weekday"]].itertuples(index=False, name=None))
                                    )

                                if dry_run:
                                    st.success("✅ Validation passed. (Dry run: no rows inserted.)")
                                else:
                                    rows = list(df[required].itertuples(index=False, name=None))
                                    execute_values(
                                        cur,
                                        """
                                        INSERT INTO factproduction
                                        (datekey, shiftkey, machinekey, operatorkey, productkey,
                                         scheduledtime_min, downtime_min, runtime_min,
                                         actualproduction, rejectcount,
                                         availability_pct, performance_pct, quality_pct, oee_pct)
                                        VALUES %s
                                        """,
                                        rows, page_size=5000
                                    )
                                    conn.commit()
                                    st.success(f"✅ Imported {len(rows):,} row(s) into factproduction.")
                                    st.toast("Import completed", icon="✅")
                            except Exception as e:
                                conn.rollback()
                                st.error(f"❌ Import failed: {e}")
                            finally:
                                cur.close()

        # ---------- Delete Data ----------
        with t2:
            st.caption("Delete by **date range**. Optionally filter by Shift/Machine/Product. This is permanent.")
            dc1, dc2 = st.columns(2)
            with dc1:
                d_from = st.date_input("From date", datetime.date.today() - datetime.timedelta(days=7), key="del_from")
            with dc2:
                d_to   = st.date_input("To date",   datetime.date.today(), key="del_to")

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_shift = st.selectbox("Filter Shift (optional)", ["-- All --"] + shifts_df["shiftname"].tolist(), key="del_shift")
            with fc2:
                f_machine = st.selectbox("Filter Machine (optional)", ["-- All --"] + machines_df["machinename"].tolist(), key="del_machine")
            with fc3:
                f_product = st.selectbox("Filter Product (optional)", ["-- All --"] + products_df["productname"].tolist(), key="del_product")

            confirm = st.text_input('Type **DELETE** to confirm', key="del_confirm")
            do_delete = st.button("Delete rows in range", type="secondary", help="Irreversible", key="del_btn")

            if do_delete:
                if confirm != "DELETE":
                    st.error("Type DELETE to confirm.")
                else:
                    datekey_from = int(d_from.strftime("%Y%m%d"))
                    datekey_to   = int(d_to.strftime("%Y%m%d"))

                    where = ["datekey BETWEEN %s AND %s"]
                    params = [datekey_from, datekey_to]

                    if f_shift != "-- All --":
                        where.append("shiftkey = %s")
                        params.append(int(shift_map[_norm(f_shift)]))
                    if f_machine != "-- All --":
                        where.append("machinekey = %s")
                        params.append(int(machine_map[_norm(f_machine)]))
                    if f_product != "-- All --":
                        where.append("productkey = %s")
                        params.append(int(product_map[_norm(f_product)]))

                    sql = "DELETE FROM factproduction WHERE " + " AND ".join(where)
                    conn = get_connection()
                    cur = conn.cursor()
                    try:
                        cur.execute(sql, params)
                        affected = cur.rowcount
                        conn.commit()
                        st.success(f"🧹 Deleted {affected:,} row(s).")
                        st.toast("Delete completed", icon="🧹")
                    except Exception as e:
                        conn.rollback()
                        st.error(f"❌ Delete failed: {e}")
                    finally:
                        cur.close()

# ---------- Right panel: Live KPIs (from current form inputs) ----------
with right:
    st.markdown("#### Today’s KPIs")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown('<div class="small-label">Availability (%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="metric-value">{locals().get("availability", 0.0):.2f}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="small-label">Performance (%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="metric-value">{locals().get("performance", 0.0):.2f}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="small-label" style="margin-top:8px;">Quality (%)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="metric-value">{locals().get("quality", 0.0):.2f}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-label" style="margin-top:8px;">OEE (%)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="metric-value">{locals().get("oee", 0.0):.2f}</div></div>', unsafe_allow_html=True)
