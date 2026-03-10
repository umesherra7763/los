"""
=============================================================
MEDICORE — BED MANAGEMENT SYSTEM
Streamlit Dashboard
=============================================================
HOW TO RUN:
    pip install streamlit pandas numpy scikit-learn plotly
    streamlit run bed_management_app.py

Put these files in the SAME folder before running:
    - bed_management_app.py        (this file)
    - model_bundle.pkl             (trained ML models)
    - synthetic_hospital_data.csv  (patient data)
    - hospital_occupancy_timeseries.csv (daily occupancy)
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="MediCore BMS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DARK THEME CSS ────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #060b14; color: #e2eaf4; }
  section[data-testid="stSidebar"] { background-color: #0d1623; border-right: 1px solid #1a2e45; }
  [data-testid="metric-container"] { background:#0d1623; border:1px solid #1a2e45; border-radius:12px; padding:16px !important; }
  [data-testid="metric-container"] label { color:#4a6080 !important; font-size:11px !important; letter-spacing:2px; text-transform:uppercase; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:32px !important; font-weight:800 !important; }
  h1,h2,h3 { color:#e2eaf4 !important; }
  .sec { font-size:12px; letter-spacing:3px; text-transform:uppercase; color:#00c8ff; font-weight:700; margin-bottom:16px; border-left:3px solid #00c8ff; padding-left:10px; }
  .bed-grid { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:8px; }
  .bed { width:30px; height:20px; border-radius:4px; border:1px solid; display:inline-block; }
  .bed-occupied    { background:#1a3a6e; border-color:#2a5098; }
  .bed-critical    { background:#5c1a1a; border-color:#ff4560; box-shadow:0 0 8px rgba(255,69,96,0.5); }
  .bed-infected    { background:#3d2a00; border-color:#ffb020; }
  .bed-surgery     { background:#0d2e3d; border-color:#00c8ff; }
  .bed-discharging { background:#0d2e1e; border-color:#00e5a0; box-shadow:0 0 8px rgba(0,229,160,0.5); }
  .bed-available   { background:#111d2e; border-color:#1a2e45; border-style:dashed; }
  .bed-predicted   { background:#2a1a5e; border-color:#a855f7; box-shadow:0 0 8px rgba(168,85,247,0.5); }
  .pcard { background:#0d1623; border:1px solid #1a2e45; border-radius:10px; padding:12px; margin-bottom:8px; }
  .leg { display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; padding-top:12px; border-top:1px solid #1a2e45; }
  .li { display:flex; align-items:center; gap:6px; font-size:11px; color:#4a6080; }
  .lb { width:16px; height:10px; border-radius:3px; border:1px solid; }
  .tag { padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; letter-spacing:1px; }
  .tc { background:rgba(255,69,96,0.15);  color:#ff4560;  border:1px solid rgba(255,69,96,0.3); }
  .ti { background:rgba(255,176,32,0.15); color:#ffb020;  border:1px solid rgba(255,176,32,0.3); }
  .ts { background:rgba(0,200,255,0.12);  color:#00c8ff;  border:1px solid rgba(0,200,255,0.3); }
  .tstb{ background:rgba(0,229,160,0.10); color:#00e5a0;  border:1px solid rgba(0,229,160,0.25);}
  .tp { background:rgba(168,85,247,0.15); color:#a855f7;  border:1px solid rgba(168,85,247,0.3);}
  .stButton button { background:linear-gradient(135deg,#0066ff,#00c8ff) !important; color:white !important; border:none !important; font-weight:700 !important; width:100% !important; padding:12px !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA & MODELS ────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model_bundle.pkl','rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_hospital_data.csv')
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    return df

@st.cache_data
def load_occ():
    return pd.read_csv('hospital_occupancy_timeseries.csv')

bundle  = load_models()
df      = load_data()
occ_df  = load_occ()
MODELS  = bundle['models']
FEATS   = bundle['feature_cols']
CCOLS   = bundle['continuous_cols']
SCALER  = bundle['scaler']

# ── CONSTANTS ─────────────────────────────────────────────────
TOTAL_BEDS = 200
TODAY      = pd.Timestamp('2026-03-10')

DEPT_BEDS = {
    'General Surgery':20,'Cardiology':18,'Orthopedic Surgery':16,
    'Obstetrics/Gynecology':14,'Neurology':12,'Gastroenterology':10,
    'Pulmonology':10,'Oncology':10,'Neurosurgery':8,'Urology':8,
    'Nephrology':8,'Endocrinology':8,'Psychiatry':8,'Pediatrics':10,
    'Rheumatology':6,'Dermatology':6,'Ophthalmology':6,'Internal Medicine':14,
    'Thoracic Surgery':8,'Hematology':8
}

DEPT_ICONS = {
    'Cardiology':'🫀','Orthopedic Surgery':'🦴','Neurology':'🧠','Neurosurgery':'🔬',
    'Hematology':'🩸','Rheumatology':'💊','General Surgery':'🏥','Nephrology':'💧',
    'Internal Medicine':'📋','Pulmonology':'🫁','Oncology':'🎗️','Gastroenterology':'🫙',
    'Obstetrics/Gynecology':'🤱','Thoracic Surgery':'⚡','Urology':'💊',
    'Endocrinology':'⚗️','Psychiatry':'🧩','Pediatrics':'👶','Dermatology':'🧴','Ophthalmology':'👁️'
}

DIAGNOSES = [
    'Cardiovascular Disease','Respiratory Disease','Gastrointestinal Disease',
    'Musculoskeletal Disease','Neurological Disease','Genitourinary Disease',
    'Endocrine Disease','Infectious Disease','Neoplasm/Cancer',
    'Trauma/Injury','Mental Health','Other'
]

MODEL_METRICS = {
    'Linear Regression':  {'MAE':1.518,'RMSE':1.905,'R²':0.460,'MAPE':'27.4%'},
    'Ridge Regression':   {'MAE':1.518,'RMSE':1.905,'R²':0.460,'MAPE':'27.4%'},
    'Decision Tree':      {'MAE':1.729,'RMSE':2.136,'R²':0.321,'MAPE':'30.6%'},
    'Random Forest':      {'MAE':1.599,'RMSE':1.992,'R²':0.410,'MAPE':'29.3%'},
    'Gradient Boosting':  {'MAE':1.560,'RMSE':1.958,'R²':0.430,'MAPE':'28.2%'},
}

# ── PERSISTENT STORAGE ────────────────────────────────────────
# Patients are saved to patients_db.json in the same folder as
# this script so data survives app restarts.

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PATIENTS_DB = os.path.join(SCRIPT_DIR, 'patients_db.json')

def _save_patients(patients: list):
    """Serialise patient list → JSON (converts Timestamps to strings)."""
    serialisable = []
    for p in patients:
        row = {}
        for k, v in p.items():
            if isinstance(v, pd.Timestamp):
                row[k] = v.strftime('%Y-%m-%d')
            else:
                row[k] = v
        serialisable.append(row)
    with open(PATIENTS_DB, 'w') as f:
        json.dump(serialisable, f, indent=2)

def _load_patients() -> list:
    """Load patient list from JSON; convert date strings back to Timestamps."""
    if not os.path.exists(PATIENTS_DB):
        return []
    with open(PATIENTS_DB, 'r') as f:
        raw = json.load(f)
    patients = []
    for row in raw:
        for date_col in ('admission_date', 'discharge_date'):
            if date_col in row and isinstance(row[date_col], str):
                row[date_col] = pd.Timestamp(row[date_col])
        patients.append(row)
    return patients

# ── SESSION STATE ─────────────────────────────────────────────
if 'pred_patients' not in st.session_state:
    # First run in this browser session — restore from disk
    st.session_state.pred_patients = _load_patients()

# ── HELPERS ───────────────────────────────────────────────────
def get_current():
    cur = df[(df['admission_date']<=TODAY)&(df['discharge_date']>TODAY)].copy()
    cur['days_in']   = (TODAY - cur['admission_date']).dt.days
    cur['days_left'] = (cur['discharge_date'] - TODAY).dt.days
    return cur

def bed_status(p):
    if p['days_left'] <= 1:             return 'discharging'
    if p['severity_of_illness'] == 3:  return 'critical'
    if p['has_infection'] == 1:        return 'infected'
    if p['surgery_performed'] == 1:    return 'surgery'
    return 'occupied'

def los_cat(d):
    if d<=3: return 'Short (1-3d)'
    elif d<=7: return 'Medium (4-7d)'
    elif d<=14: return 'Long (8-14d)'
    else: return 'Very Long (>14d)'

def predict_los(inp, model_name):
    row = {c:0 for c in FEATS}
    row['age']                        = inp['age']
    row['gender']                     = 1 if inp['gender']=='Female' else 0
    row['admission_type']             = 1 if inp['admission_type']=='Emergency' else 0
    row['severity_of_illness']        = inp['severity']
    row['has_comorbidity']            = inp['comorbidity']
    row['charlson_comorbidity_index'] = inp['charlson']
    row['has_infection']              = inp['infection']
    row['surgery_performed']          = inp['surgery']
    row['surgery_duration_mins']      = inp['surg_dur']
    row['num_procedures']             = inp['n_proc']
    row['num_medications']            = inp['n_meds']
    row['weight_loss']                = inp['wt_loss']
    row['dietary_pattern_change']     = inp['diet']
    row['functional_capacity_change'] = inp['func']
    row['malnutrition']               = inp['malnut']
    row['bmi']                        = inp['bmi']
    row['systolic_bp']                = inp['sbp']
    row['diastolic_bp']               = inp['dbp']
    row['body_temperature']           = inp['temp']
    row['blood_glucose']              = inp['glucose']
    row['previous_admissions']        = inp['prev_adm']
    row['outpatient_visits_30d']      = inp['op_visits']
    row['admission_month']            = inp['adm_date'].month
    row['admission_year']             = inp['adm_date'].year
    row['weekend_admission']          = 1 if inp['adm_date'].weekday()>=5 else 0
    row['hour_of_admission']          = inp['hour']
    row['insurance_type']             = {'Uninsured':0,'Government':1,'Private':2}[inp['insurance']]
    row['socioeconomic_status']       = {'Low':0,'Middle':1,'High':2}[inp['ses']]
    row['marital_status']             = {'Single':0,'Married':1,'Divorced':2,'Widowed':3}[inp['marital']]

    X = pd.DataFrame([row])[FEATS]
    X[CCOLS] = SCALER.transform(X[CCOLS])
    pred = MODELS[model_name].predict(X)[0]
    return int(np.clip(round(pred), 1, 30))


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 MediCore BMS")
    st.caption("Bed Management System · ML-Powered")
    st.divider()

    sel_model = st.selectbox("🤖 ML Model", list(MODELS.keys()))
    st.divider()

    st.markdown("### ➕ Admit New Patient")
    st.caption("Fill in details → Predict LOS → Assign Bed")

    with st.expander("👤 Demographics", expanded=True):
        age     = st.slider("Age", 18, 90, 50)
        gender  = st.selectbox("Gender", ["Male","Female"])
        marital = st.selectbox("Marital Status", ["Single","Married","Divorced","Widowed"])
        ses     = st.selectbox("Socioeconomic Status", ["Low","Middle","High"])
        ins     = st.selectbox("Insurance", ["Government","Private","Uninsured"])

    with st.expander("🏨 Admission", expanded=True):
        dept     = st.selectbox("Department", list(DEPT_BEDS.keys()))
        adm_type = st.selectbox("Admission Type", ["Elective","Emergency"])
        adm_date = st.date_input("Admission Date", datetime(2024,1,1))
        hour     = st.selectbox("Hour", [0,1,2,3], format_func=lambda x:["09-12","12-15","15-18","18-21"][x])

    with st.expander("🩺 Clinical", expanded=True):
        severity   = st.slider("Severity (1=Mild, 3=Severe)", 1, 3, 2)
        comorbidity= st.checkbox("Has Comorbidity")
        charlson   = st.slider("Charlson Index", 0, 10, 0, disabled=not comorbidity)
        infection  = st.checkbox("Has Infection")
        malnut     = st.checkbox("Malnutrition")
        wt_loss    = st.checkbox("Weight Loss")
        diet       = st.checkbox("Dietary Pattern Change")
        func       = st.checkbox("Functional Capacity Change")

    with st.expander("📊 Vitals"):
        c1,c2 = st.columns(2)
        sbp    = c1.number_input("Systolic BP",  80, 200, 120)
        dbp    = c2.number_input("Diastolic BP", 50, 130,  80)
        temp   = c1.number_input("Temp °C", 35.0, 42.0, 37.0, 0.1)
        gluc   = c2.number_input("Glucose",  60,  400, 100)
        bmi    = st.number_input("BMI", 15.0, 50.0, 25.0, 0.5)

    with st.expander("🔪 Surgery & History"):
        surgery  = st.checkbox("Surgery Performed")
        surg_dur = st.slider("Surgery Duration (mins)", 0, 480, 0, disabled=not surgery)
        n_proc   = st.slider("# Procedures", 0, 10, 1)
        n_meds   = st.slider("# Medications", 0, 20, 3)
        prev_adm = st.slider("Previous Admissions", 0, 10, 0)
        op_vis   = st.slider("Outpatient Visits (30d)", 0, 15, 2)

    st.divider()
    predict_btn = st.button("🤖 PREDICT LOS & ADMIT", use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════
new_pred = None
if predict_btn:
    inp = {
        'age':age,'gender':gender,'marital':marital,'ses':ses,'insurance':ins,
        'admission_type':adm_type,'adm_date':adm_date,'hour':hour,
        'severity':severity,'comorbidity':int(comorbidity),'charlson':charlson if comorbidity else 0,
        'infection':int(infection),'surgery':int(surgery),
        'surg_dur':surg_dur if surgery else 0,'n_proc':n_proc,'n_meds':n_meds,
        'wt_loss':int(wt_loss),'diet':int(diet),'func':int(func),'malnut':int(malnut),
        'bmi':bmi,'sbp':sbp,'dbp':dbp,'temp':temp,'glucose':gluc,
        'prev_adm':prev_adm,'op_visits':op_vis,
    }
    los = predict_los(inp, sel_model)
    disc_date = pd.Timestamp(adm_date) + timedelta(days=los)

    pstatus = 'critical' if severity==3 else 'infected' if infection else 'surgery' if surgery else 'occupied'

    new_pred = {
        'patient_id': f"NEW-{len(st.session_state.pred_patients)+1:03d}",
        'age':age,'gender':gender,'department':dept,
        'admission_date':pd.Timestamp(adm_date),
        'discharge_date':disc_date,
        'length_of_stay':los,'los_category':los_cat(los),
        'severity_of_illness':severity,
        'has_infection':int(infection),'surgery_performed':int(surgery),
        'status':pstatus,'days_left':los,'days_in':0,
        'model_used':sel_model,
    }
    st.session_state.pred_patients.append(new_pred)
    _save_patients(st.session_state.pred_patients)


# ═════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            background:#0d1623;border:1px solid #1a2e45;border-radius:14px;
            padding:18px 24px;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <span style="font-size:32px">🏥</span>
    <div>
      <div style="font-size:20px;font-weight:900">MediCore BMS</div>
      <div style="font-size:11px;color:#4a6080;letter-spacing:2px;text-transform:uppercase">Bed Management System · ML-Powered LOS Prediction</div>
    </div>
  </div>
  <div style="display:flex;gap:12px;align-items:center;">
    <div style="background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.3);border-radius:20px;padding:6px 14px;font-size:11px;color:#00e5a0">● LIVE</div>
    <div style="font-size:12px;color:#4a6080;font-family:monospace">10 MAR 2026</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Prediction result banner
if new_pred:
    d = new_pred
    cat_color = '#00e5a0' if 'Short' in d['los_category'] else '#ffb020' if 'Medium' in d['los_category'] else '#ff4560'
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2e3d,#082030);border:2px solid #00c8ff;
                border-radius:14px;padding:24px;margin-bottom:20px;
                display:flex;align-items:center;gap:32px;">
      <div style="text-align:center;min-width:100px;">
        <div style="font-size:56px;font-weight:900;color:#00c8ff;line-height:1">{d['length_of_stay']}</div>
        <div style="font-size:10px;color:#4a6080;letter-spacing:2px;text-transform:uppercase;margin-top:4px">DAYS PREDICTED</div>
      </div>
      <div>
        <div style="font-size:15px;font-weight:700;margin-bottom:10px">
          ★ Patient <span style="color:#a855f7">{d['patient_id']}</span> admitted to
          <span style="color:#00c8ff">{d['department']}</span>
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;">
          <span style="font-size:12px;color:#4a6080">📅 Discharge: <b style="color:#00e5a0">{d['discharge_date'].strftime('%d %b %Y')}</b></span>
          <span style="font-size:12px;color:#4a6080">📊 Category: <b style="color:{cat_color}">{d['los_category']}</b></span>
          <span style="font-size:12px;color:#4a6080">🤖 Model: <b style="color:#a855f7">{d['model_used']}</b></span>
        </div>
        <div style="margin-top:8px;font-size:11px;color:#4a6080">
          Bed occupied for <b style="color:#e2eaf4">{d['length_of_stay']} days</b> ·
          Expected departure <b style="color:#00e5a0">{d['discharge_date'].strftime('%A, %d %b')}</b>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# KPIs
cur = get_current()
n_occ  = len(cur) + len(st.session_state.pred_patients)
n_avail= TOTAL_BEDS - n_occ
n_crit = int((cur['severity_of_illness']==3).sum())
n_disc_today = int((cur['days_left']<=1).sum())

st.markdown('<div class="sec">Real-Time Overview</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("🛏️ Total Beds",       TOTAL_BEDS)
c2.metric("🔴 Occupied",          n_occ,   delta=f"+{len(st.session_state.pred_patients)} new" if st.session_state.pred_patients else None)
c3.metric("🟢 Available",         n_avail, delta=f"-{len(st.session_state.pred_patients)}" if st.session_state.pred_patients else None, delta_color="inverse")
c4.metric("⚡ Critical",           n_crit)
c5.metric("🚪 Discharging Today", n_disc_today)

st.divider()

# Main 2-col layout
left, right = st.columns([2.2, 1])

# ── LEFT: BED MAP ─────────────────────────────────────────────
with left:
    st.markdown('<div class="sec">Live Bed Map — All Departments</div>', unsafe_allow_html=True)

    # Build dept → patients lookup
    dp = {}
    for _, p in cur.iterrows():
        d = p['department']
        dp.setdefault(d,[]).append({
            'id':p['patient_id'],'age':int(p['age']),'days_in':int(p['days_in']),
            'days_left':int(p['days_left']),'status':bed_status(p),'predicted':False
        })
    for pp in st.session_state.pred_patients:
        dp.setdefault(pp['department'],[]).append({
            'id':pp['patient_id'],'age':pp['age'],'days_in':0,
            'days_left':pp['days_left'],'status':'predicted','predicted':True
        })

    for dname, dtotal in DEPT_BEDS.items():
        pts   = dp.get(dname,[])
        n_occ_dept = len(pts)
        pct   = round(n_occ_dept/dtotal*100)
        icon  = DEPT_ICONS.get(dname,'🏥')
        label = f"{icon} {dname}  ·  {n_occ_dept}/{dtotal}  ({pct}% occupied)"

        with st.expander(label, expanded=n_occ_dept>0):
            html = '<div class="bed-grid">'
            for p in pts:
                tip = f"{p['id']} | {p['age']}y | Day {p['days_in']} | {p['days_left']}d left"
                if p['predicted']: tip += " | ★ ML Predicted"
                html += f'<div class="bed bed-{p["status"]}" title="{tip}"></div>'
            for _ in range(dtotal - n_occ_dept):
                html += '<div class="bed bed-available" title="Available"></div>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("""
    <div class="leg">
      <div class="li"><div class="lb" style="background:#5c1a1a;border-color:#ff4560"></div>Critical</div>
      <div class="li"><div class="lb" style="background:#3d2a00;border-color:#ffb020"></div>Infected</div>
      <div class="li"><div class="lb" style="background:#0d2e3d;border-color:#00c8ff"></div>Post-Surgery</div>
      <div class="li"><div class="lb" style="background:#0d2e1e;border-color:#00e5a0"></div>Discharging ≤1d</div>
      <div class="li"><div class="lb" style="background:#1a3a6e;border-color:#2a5098"></div>Stable</div>
      <div class="li"><div class="lb" style="background:#2a1a5e;border-color:#a855f7"></div>★ ML Predicted</div>
      <div class="li"><div class="lb" style="background:#111d2e;border-color:#1a2e45;border-style:dashed"></div>Available</div>
    </div>""", unsafe_allow_html=True)

# ── RIGHT: DISCHARGE + METRICS ────────────────────────────────
with right:
    st.markdown('<div class="sec">Discharging Soon</div>', unsafe_allow_html=True)

    all_pts = cur[['patient_id','age','gender','department','days_left',
                   'severity_of_illness','has_infection','surgery_performed']].copy()
    for pp in st.session_state.pred_patients:
        all_pts = pd.concat([all_pts, pd.DataFrame([{
            'patient_id':pp['patient_id'],'age':pp['age'],'gender':pp['gender'],
            'department':pp['department'],'days_left':pp['days_left'],
            'severity_of_illness':pp['severity_of_illness'],
            'has_infection':pp['has_infection'],'surgery_performed':pp['surgery_performed'],
        }])], ignore_index=True)

    cards_html = ""
    for _, p in all_pts.sort_values('days_left').head(10).iterrows():
        dl = int(p['days_left'])
        day_str = "TODAY" if dl<=0 else "TOMORROW" if dl==1 else f"{dl} DAYS"
        day_col = '#00e5a0' if dl<=1 else '#ffb020' if dl<=3 else '#e2eaf4'
        sev = int(p['severity_of_illness'])
        tag_cls = 'tc' if sev==3 else ('ti' if int(p['has_infection'])==1 else ('ts' if int(p['surgery_performed'])==1 else 'tstb'))
        tag_lbl = 'CRITICAL' if sev==3 else ('INFECTED' if int(p['has_infection'])==1 else ('POST-OP' if int(p['surgery_performed'])==1 else 'STABLE'))
        ml_badge = '<span class="tag tp">&#9733; ML</span>' if str(p['patient_id']).startswith('NEW-') else ''
        cards_html += (
            f'<div class="pcard">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="font-family:monospace;font-size:11px;color:#00c8ff">{p["patient_id"]}</span>'
            f'{ml_badge}'
            f'<span style="font-size:16px;font-weight:900;color:{day_col}">{day_str}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#4a6080;margin-bottom:6px">{p["gender"]} &middot; {int(p["age"])}y &middot; {str(p["department"])[:22]}</div>'
            f'<span class="tag {tag_cls}">{tag_lbl}</span>'
            f'</div>'
        )
    st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec">ML Model Performance</div>', unsafe_allow_html=True)
    m = MODEL_METRICS[sel_model]
    mc1,mc2 = st.columns(2)
    mc1.metric("MAE",  f"{m['MAE']}d")
    mc2.metric("R²",   m['R²'])
    mc3,mc4 = st.columns(2)
    mc3.metric("RMSE", m['RMSE'])
    mc4.metric("MAPE", m['MAPE'])
    st.markdown(f"""
    <div style="background:rgba(0,229,160,0.06);border:1px solid rgba(0,229,160,0.2);
                border-radius:8px;padding:10px 12px;margin-top:8px;">
      <div style="font-size:10px;color:#00e5a0;letter-spacing:1px">ACTIVE MODEL</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">{sel_model}</div>
      <div style="font-size:10px;color:#4a6080">Avg error: ±{m['MAE']} days</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── CHARTS ────────────────────────────────────────────────────
st.markdown('<div class="sec">Analytics & Trends</div>', unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)

with g1:
    trend = occ_df.tail(14).copy()
    fig = go.Figure(go.Bar(
        x=trend['date'], y=trend['patients_in_hospital'],
        marker=dict(color=trend['patients_in_hospital'],
                    colorscale=[[0,'#0066ff'],[0.5,'#00c8ff'],[1,'#00e5a0']]),
    ))
    fig.update_layout(
        title='14-Day Occupancy Trend',
        paper_bgcolor='#0d1623',plot_bgcolor='#0d1623',
        font=dict(color='#e2eaf4',size=11),height=250,
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(gridcolor='#1a2e45',tickfont=dict(size=8)),
        yaxis=dict(gridcolor='#1a2e45',title='Patients'),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with g2:
    dept_data = [(d,len(dp.get(d,[])),t) for d,t in DEPT_BEDS.items() if len(dp.get(d,[]))>0]
    dept_data.sort(key=lambda x:-x[1])
    fig2 = go.Figure(go.Bar(
        x=[x[1] for x in dept_data], y=[x[0] for x in dept_data], orientation='h',
        marker=dict(color=[x[1]/x[2] for x in dept_data],
                    colorscale=[[0,'#0066ff'],[0.5,'#ffb020'],[1,'#ff4560']],showscale=False),
        text=[f"{x[1]}/{x[2]}" for x in dept_data],
        textposition='outside', textfont=dict(size=9,color='#e2eaf4')
    ))
    fig2.update_layout(
        title='Department Occupancy',
        paper_bgcolor='#0d1623',plot_bgcolor='#0d1623',
        font=dict(color='#e2eaf4',size=10),height=250,
        margin=dict(l=10,r=40,t=40,b=10),
        xaxis=dict(gridcolor='#1a2e45'),
        yaxis=dict(gridcolor='#1a2e45',tickfont=dict(size=9)),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

with g3:
    fc = []
    for i in range(1,8):
        fut = TODAY + timedelta(days=i)
        d = int((df['discharge_date']==fut).sum())
        for pp in st.session_state.pred_patients:
            if pp['discharge_date']==fut: d+=1
        fc.append({'date':fut.strftime('%b %d'),'discharges':d})
    fc_df = pd.DataFrame(fc)
    fig3 = go.Figure(go.Scatter(
        x=fc_df['date'], y=fc_df['discharges'], fill='tozeroy',
        line=dict(color='#00e5a0',width=2), fillcolor='rgba(0,229,160,0.1)',
        mode='lines+markers', marker=dict(color='#00e5a0',size=8)
    ))
    fig3.update_layout(
        title='Discharge Forecast (7 Days)',
        paper_bgcolor='#0d1623',plot_bgcolor='#0d1623',
        font=dict(color='#e2eaf4',size=11),height=250,
        margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(gridcolor='#1a2e45'),
        yaxis=dict(gridcolor='#1a2e45',title='Discharges'),
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── PREDICTED PATIENTS TABLE ──────────────────────────────────
if st.session_state.pred_patients:
    st.divider()
    st.markdown('<div class="sec">★ ML-Predicted Patients This Session</div>', unsafe_allow_html=True)
    tdf = pd.DataFrame(st.session_state.pred_patients)[[
        'patient_id','age','gender','department','admission_date',
        'discharge_date','length_of_stay','los_category','model_used'
    ]]
    tdf.columns = ['Patient','Age','Gender','Department','Admitted','Discharges','LOS (days)','Category','Model']
    st.dataframe(tdf, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear Predicted Patients"):
        st.session_state.pred_patients = []
        _save_patients([])   # wipe the file too
        st.rerun()

st.divider()
st.markdown('<div style="text-align:center;color:#1a2e45;font-size:11px">MediCore BMS · ML-Powered Bed Management · Hospital LOS Prediction System</div>', unsafe_allow_html=True)
