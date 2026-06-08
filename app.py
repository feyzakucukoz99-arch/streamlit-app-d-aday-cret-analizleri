import streamlit as st
import pandas as pd
import numpy as np
import os, unicodedata, smtplib, tempfile, io, re
from io import BytesIO
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__VERSION__ = "2025-10-10.r28 (Kıdem kolonu ve e-posta ayarları koda gömüldü - GÜVENLİK UYARISI)"

# ===================== UI =====================
st.set_page_config(page_title="⚡ Dış Aday Teklif Asistanı", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"]{color:#1976D2}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{font-size:20px}
div[data-testid="stMetric"] div[data-testid="stMetricLabel"]{font-size:12px}
.eq-badge{background:#E3F2FD;border:1px solid #64B5F6;color:#1976D2;padding:4px 8px;border-radius:12px;font-size:.85rem}
.warn{background:#FFF8E1;border-left:4px solid #FF9800;padding:12px}
.ok{background:#E8F5E9;border-left:4px solid #43A047;padding:12px}
.card{border:1px solid #eee;padding:14px;border-radius:12px;background:#fff}
.title-accent{color:#1976D2}
.small{font-size:.9rem;color:#4b5563}
/* sohbet tek kutu */
.chatbox{border:1px solid #e5e7eb;border-radius:14px;padding:10px;background:#fff;max-width:700px;margin:0 auto}
.chatlog{max-height:360px;overflow-y:auto;padding:6px;background:#fafafa;border-radius:10px;border:1px solid #eee}
.msgu{background:#E3F2FD;padding:8px;border-radius:10px;margin:6px 0}
.msga{background:#F1F8E9;padding:8px;border-radius:10px;margin:6px 0}
</style>
""", unsafe_allow_html=True)
st.title("⚡ Dış Aday Teklif Asistanı")
st.caption(f"Sürüm: {__VERSION__}")

# ===================== Helpers =====================
def tr_fold(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = s.translate(str.maketrans("çğıöşüÇĞİÖŞÜ","cgiosuCGIOSU"))
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())

ALIASES = {
    "pozisyon":["pozisyon","harmony pozisyon","pozisyon adi"],
    "ucret":["ucret","ücret","ucret (tl)","ücret (tl)","ÜCRET"],
    "kidem_total":["harici+dahili kıdem","harici+dahili kidem","şirket total kıdem","sirket total kidem","harici dahili kidem", "kıdem", "kidem"], # "kıdem" ve "kidem" eklendi
    "kidem_poz":["son pozisyon kıdem","pozisyon kidem"],
    "ise_giris":["işe giriş tarihi","ise giris tarihi","ise giris"],
    "skala_medyan":["skala medyanı","skala medyani"],
    "skala_min":["pozisyon bazında minimum ücret","pozisyon bazinda minimum ucret","pozisyon bazinda min ucret"],
    "skala_max":["pozisyon bazında maksimum ücret","pozisyon bazinda maksimum ucret","pozisyon bazinda max ucret"],
    "ogren_bas":["öğrenen başlangıç","ogrenen baslangic"],
    "yuksek_bitis":["yüksek bitiş","yuksek bitis"],
    "band_konum":["ücret konumu","ucret konumu"],
    "yon1":["1. yöneticisi","1. yoneticisi","birinci yoneticisi"],
    "yon2":["2. yöneticisi","2. yoneticisi","ikinci yoneticisi"],
    "perf":["şubat 2025 performans sonuçları","subat 2025 performans sonuclari","şubat 2025 performans","subat 2025 performans"],
    "pr":["kişi pr","kisi pr","pr","kişipr","kisi_pr"],
    "artis_oran":["şubat 2025 artış oranı","subat 2025 artis orani"],
}

# Sütun adlarını çözümlemek için iç yardımcı fonksiyon
def _resolve_col_helper(df_folded_cols: dict, key: str, required=False):
    for alias in ALIASES.get(key, []):
        if alias in df_folded_cols:
            return df_folded_cols[alias]
    if required:
        raise KeyError(f"Gerekli sütun bulunamadı: {key}")
    return None

def money_series_to_float_vectorized_robust(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace("", np.nan)

    s_cleaned_symbols = s.str.replace('TL', '', regex=False) \
                         .str.replace('₺', '', regex=False) \
                         .str.replace('TRY', '', regex=False) \
                         .str.replace('\u00A0', '', regex=False)
    s_cleaned_symbols = s_cleaned_symbols.str.replace(' ', '', regex=False)

    def parse_money_string_single(num_str):
        if pd.isna(num_str) or not isinstance(num_str, str):
            return np.nan
        num_str = num_str.strip()
        if not num_str:
            return np.nan

        last_comma_idx = num_str.rfind(',')
        last_dot_idx = num_str.rfind('.')

        if last_comma_idx > last_dot_idx:
            num_str = num_str.replace('.', '').replace(',', '.')
        elif last_dot_idx > last_comma_idx:
            num_str = num_str.replace(',', '')
        else:
            if ',' in num_str:
                num_str = num_str.replace(',', '')
            else:
                pass

        try:
            return float(num_str)
        except ValueError:
            return np.nan

    return s_cleaned_symbols.apply(parse_money_string_single)


def percentiles_num(arr):
    a = pd.to_numeric(arr, errors="coerce").dropna().values
    if a.size==0: return (np.nan, np.nan, np.nan, np.nan, np.nan)
    return (np.nanpercentile(a,25), np.nanmedian(a), np.nanpercentile(a,75), a.min(), a.max())

def exp_to_quantile(series_exp, used_exp):
    if used_exp is None or pd.isna(used_exp): return 0.50
    s = pd.to_numeric(series_exp, errors="coerce").dropna().values
    if s.size>=8:
        p25, p75 = np.nanpercentile(s,[25,75]) 
        lo,hi = (min(s),max(s)) if p25==p75 else (p25,p75)
        if lo!=hi:
            q = (used_exp - lo)/(hi-lo)
            return float(np.clip(0.1 + 0.8*q, 0.0, 1.0))
    return 0.25 if used_exp<2 else 0.50 if used_exp<4 else 0.70 if used_exp<6 else 0.85

def suggest_range_from_scale(smin,smed,smax, sal_arr, used_q, max_width=10_000):
    lo = np.nanmin([smin,smax]); hi = np.nanmax([smin,smax])
    have_scale = np.isfinite(lo) and np.isfinite(hi) and lo < hi
    if have_scale and np.isfinite(used_q):
        target = lo + used_q*(hi-lo); band_lo, band_hi = lo, hi
    else:
        if sal_arr.size>=4:
            p25,p50,p75 = np.nanpercentile(sal_arr,[25,50,75])
            target = np.interp(used_q if np.isfinite(used_q) else 0.5, [0,0.5,1], [p25,p50,p75])
            band_lo, band_hi = p25, p75
        else:
            target = smed; band_lo, band_hi = (smed*0.9, smed*1.1) if np.isfinite(smed) else (np.nan, np.nan)
    if np.isfinite(band_lo) and np.isfinite(band_hi) and (band_hi - band_lo) > max_width:
        center = target if np.isfinite(target) else ((band_lo+band_hi)/2 if np.isfinite(band_lo) and np.isfinite(band_hi) else smed)
        band_lo = max(0, center - max_width/2); band_hi = band_lo + max_width
    if have_scale:
        if np.isfinite(target): target = float(np.clip(target, lo, hi))
        if np.isfinite(band_lo): band_lo = float(max(band_lo, lo))
        if np.isfinite(band_hi): band_hi = float(min(band_hi, hi))
    width = (band_hi - band_lo) if (np.isfinite(band_hi) and np.isfinite(band_lo)) else 0
    if (not np.isfinite(width)) or width <= 0:
        band_lo = max(lo, target - max_width/2); band_hi = min(hi, band_lo + max_width)
    if band_hi - band_lo < 0: band_lo, band_hi = lo, hi
    return float(target), (float(band_lo), float(band_hi))

def build_display(df, sal_col, kid_total_col=None, hire_col=None, perf_col=None, artis_oran=None, refs=None, mgr_cols=None, extra_cols=None):
    if df is None or df.empty: return pd.DataFrame()
    cols = []
    if refs: cols += [r for r in refs if r in df.columns]
    cols.append(sal_col)
    if kid_total_col: cols.append(kid_total_col)
    if hire_col: cols.append(hire_col)
    if perf_col: cols.append(perf_col)
    if artis_oran: cols.append(artis_oran)
    if mgr_cols: cols += [c for c in mgr_cols if c in df.columns]
    if extra_cols: cols += [c for c in extra_cols if c and c in df.columns]
    view = df[[c for c in cols if c in df.columns]].copy()
    ren = {sal_col:"Ücret"}
    if kid_total_col: ren[kid_total_col] = "Harici+Dahili Kıdem"
    if hire_col: ren[hire_col] = "İşe Giriş"
    if perf_col: ren[perf_col] = "ŞUBAT 2025 PERFORMANS SONUÇLARI"
    if artis_oran: ren[artis_oran] = "ŞUBAT 2025 ARTIŞ ORANI"
    view.rename(columns=ren, inplace=True)
    if "İşe Giriş" in view.columns:
        view["İşe Giriş"] = pd.to_datetime(view["İşe Giriş"], errors="coerce", cache=True).dt.date
    return view

# ===================== Sidebar =====================
with st.sidebar:
    try:
        st.info(f"Pandas sürümü: {pd.__version__}")
    except Exception as e:
        st.error(f"Pandas sürümü kontrol edilemedi: {e}")
    st.markdown("---")

    uploaded = st.file_uploader("Ücret listesi (Excel)", type=["xlsx","xls"])
    pos = st.text_input("Pozisyon", placeholder="Örn. Buyer")
    proposed = st.number_input("Önerdiğin Teklif (TL)", min_value=0, step=1000)
    cand_total_exp = st.number_input("Aday Toplam İş Deneyimi (yıl)", min_value=0.0, step=0.5, value=0.0)
    cand_rel_exp = st.number_input("Aday Pozisyona İlişkin Deneyim (yıl)", min_value=0.0, step=0.5, value=0.0)
    eq_threshold = st.slider("Eşitlik Uyarı Eşiği (%)", 0, 30, 5)
    st.markdown("---")
    st.markdown("### 👥 Yönetici Filtresi")
    st.session_state.setdefault("mgr_selection", [])
    st.session_state.setdefault("include_sub", True)
    
    if st.button("🔄 Veriyi yenile"):
        st.cache_data.clear(); st.rerun()

# ===================== Data Load & Initial Processing (Cached) =====================
def _file_key_from_bytes(b: bytes):
    import hashlib as _h; return "B|" + _h.md5(b).hexdigest()

@st.cache_data(show_spinner="Veri yükleniyor ve işleniyor...", ttl=1800)
def _read_and_process_excel_from_bytes(bkey, raw_bytes):
    try:
        bio = BytesIO(raw_bytes)
        sheets = pd.read_excel(bio, sheet_name=None)
        
        df = sheets["Veriler"].copy() if "Veriler" in sheets else sheets[list(sheets.keys())[0]].copy()

        df_folded_columns_map = {tr_fold(c): c for c in df.columns}

        resolved_cols_names = {}
        for key in ALIASES.keys():
            resolved_cols_names[key] = _resolve_col_helper(df_folded_columns_map, key)
        resolved_cols_names["pos_col"] = _resolve_col_helper(df_folded_columns_map, "pozisyon", required=True)
        resolved_cols_names["sal_col"] = _resolve_col_helper(df_folded_columns_map, "ucret", required=True)

        df[resolved_cols_names["sal_col"]] = money_series_to_float_vectorized_robust(df[resolved_cols_names["sal_col"]])
        for c_key in ["skala_medyan", "skala_min", "skala_max", "ogren_bas", "yuksek_bitis"]:
            c = resolved_cols_names[c_key]
            if c: df[c] = money_series_to_float_vectorized_robust(df[c])

        if resolved_cols_names["pr"]:
            def to_pr_optimized_internal(series: pd.Series) -> pd.Series:
                s = series.astype(str).str.strip().str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
                v = pd.to_numeric(s, errors="coerce")
                return np.where((v <= 1) & v.notna(), v * 100, v)
            df[resolved_cols_names["pr"]] = to_pr_optimized_internal(df[resolved_cols_names["pr"]])

        if resolved_cols_names["ise_giris"]:
            df[resolved_cols_names["ise_giris"]] = pd.to_datetime(df[resolved_cols_names["ise_giris"]], errors="coerce", cache=True)

        pos_cols_all_df = [c for c in df.columns if "pozisyon" in tr_fold(c)]
        if not pos_cols_all_df: pos_cols_all_df = [resolved_cols_names["pos_col"]]
        for i, c in enumerate(pos_cols_all_df):
            keyname = f"_pos_key_{i}"
            df[keyname] = df[c].astype(str).map(tr_fold)

        return df, resolved_cols_names
    except Exception as e:
        st.error(f"Veri işleme sırasında bir hata oluştu: {e}")
        raise

_df_processed = None
resolved_cols = None
_df_raw_cached_key = None

if uploaded is not None:
    raw = uploaded.getvalue()
    _df_raw_cached_key = _file_key_from_bytes(raw)
    _df_processed, resolved_cols = _read_and_process_excel_from_bytes(_df_raw_cached_key, raw)
else:
    default_path = os.path.join(os.getcwd(), "ücret listesi.xlsx")
    if os.path.exists(default_path):
        with open(default_path, "rb") as f:
            default_raw = f.read()
            _df_raw_cached_key = _file_key_from_bytes(default_raw)
            _df_processed, resolved_cols = _read_and_process_excel_from_bytes(_df_raw_cached_key, default_raw)
    else:
        st.info("Soldan dosya yükleyin ya da klasöre 'ücret listesi.xlsx' koyun."); st.stop()

pos_col = resolved_cols["pos_col"]
sal_col = resolved_cols["sal_col"]
kid_total_col = resolved_cols["kidem_total"]
kid_poz_col = resolved_cols["kidem_poz"]
hire_col = resolved_cols["ise_giris"]
sk_med_col = resolved_cols["skala_medyan"]
sk_min_col = resolved_cols["skala_min"]
sk_max_col = resolved_cols["skala_max"]
ogren_bas_col = resolved_cols["ogren_bas"]
yuksek_bit_col= resolved_cols["yuksek_bitis"]
band_konum = resolved_cols["band_konum"]
perf_col = resolved_cols["perf"]
pr_col = resolved_cols["pr"]
artis_oran = resolved_cols["artis_oran"]
yon1_col = resolved_cols["yon1"]
yon2_col = resolved_cols["yon2"]

# ===================== Filtering (Cached) =====================
@st.cache_data(show_spinner="Veriler filtreleniyor...", ttl=300)
def get_filtered_dfs(processed_df_cache_key, processed_df_data, pos_filter, include_sub_filter, manager_selection_list, resolved_cols_dict):
    df_copy = processed_df_data.copy()

    pkey = tr_fold(pos_filter)
    masks = []
    pos_key_cols = [c for c in df_copy.columns if c.startswith("_pos_key_")]
    for keyname in pos_key_cols:
        col = df_copy[keyname]
        masks.append(col.str.contains(pkey, na=False) if include_sub_filter else (col == pkey))
    mask_pos = np.logical_or.reduce(masks) if masks else np.array([False]*len(df_copy))

    df_pos_all_filtered = df_copy[mask_pos].copy()

    df_pos_filtered = df_pos_all_filtered.copy()
    if manager_selection_list:
        mm = []
        yon1_col_name = resolved_cols_dict["yon1"]
        yon2_col_name = resolved_cols_dict["yon2"]
        if yon1_col_name and yon1_col_name in df_pos_filtered.columns:
            mm.append(df_pos_filtered[yon1_col_name].astype(str).isin(manager_selection_list))
        if yon2_col_name and yon2_col_name in df_pos_filtered.columns:
            mm.append(df_pos_filtered[yon2_col_name].astype(str).isin(manager_selection_list))
        if mm:
            df_pos_filtered = df_pos_filtered[np.logical_or.reduce(mm)].copy()

    return df_pos_all_filtered, df_pos_filtered

if not pos:
    st.warning("Pozisyon giriniz.")
    st.stop()
df_pos_all, df_pos = get_filtered_dfs(
    _df_raw_cached_key,
    _df_processed,
    pos,
    st.session_state["include_sub"],
    st.session_state["mgr_selection"],
    resolved_cols
)

if df_pos_all.empty:
    st.error("Bu pozisyon için veri bulunamadı."); st.stop()

mgr_opts = []
for c in [yon1_col, yon2_col]:
    if c and c in df_pos_all.columns:
        mgr_opts += list(df_pos_all[c].dropna().astype(str).unique())
mgr_opts = sorted({m for m in mgr_opts if str(m).strip() != ""})

with st.sidebar:
    st.session_state["include_sub"] = st.checkbox("Alt/benzer unvanları da dahil et (içeren arama)", value=st.session_state["include_sub"])

    if mgr_opts:
        st.session_state["mgr_selection"] = st.multiselect("Yönetici (tek filtre)", mgr_opts, default=st.session_state["mgr_selection"])
        if st.button("Yönetici filtresini temizle"):
            st.session_state["mgr_selection"] = []
            st.rerun()
    else:
        st.session_state["mgr_selection"] = []
        st.info("Bu pozisyon için yönetici bulunamadı.")

st.caption(f"Filtrelenen kayıt sayısı: {len(df_pos)} (pozisyona uygun toplam: {len(df_pos_all)}){ ' — Yönetici filtresi aktif' if st.session_state['mgr_selection'] else '' }")

# ===================== Scale & distribution (Cached) =====================
@st.cache_data(show_spinner=False, ttl=300)
def get_scale_medians_cached(processed_df_cache_key, pos_filter, include_sub_filter, manager_selection_list, sal_col, ogren_bas_col, sk_min_col, sk_max_col, yuksek_bit_col, sk_med_col, resolved_cols_dict):
    _, df_pos_for_cache = get_filtered_dfs(processed_df_cache_key, _df_processed, pos_filter, include_sub_filter, manager_selection_list, resolved_cols_dict)

    scmin_r = float(pd.to_numeric(df_pos_for_cache[ogren_bas_col], errors="coerce").dropna().median()) if ogren_bas_col and df_pos_for_cache[ogren_bas_col].notna().any() else (
        float(pd.to_numeric(df_pos_for_cache[sk_min_col], errors="coerce").dropna().median()) if sk_min_col and df_pos_for_cache[sk_min_col].notna().any() else np.nan
    )
    scmax_r = float(pd.to_numeric(df_pos_for_cache[yuksek_bit_col],errors="coerce").dropna().median()) if yuksek_bit_col and df_pos_for_cache[yuksek_bit_col].notna().any() else (
        float(pd.to_numeric(df_pos_for_cache[sk_max_col], errors="coerce").dropna().median()) if sk_max_col and df_pos_for_cache[sk_max_col].notna().any() else np.nan
    )
    scmed_r = float(pd.to_numeric(df_pos_for_cache[sk_med_col], errors="coerce").dropna().median()) if sk_med_col and df_pos_for_cache[sk_med_col].notna().any() else np.nan

    return percentiles_num(df_pos_for_cache[sal_col]) + (scmin_r, scmed_r, scmax_r)


p25,p50,p75,amin,amax,scmin,scmed,scmax = get_scale_medians_cached(
    _df_raw_cached_key, pos, st.session_state["include_sub"], st.session_state["mgr_selection"],
    sal_col, ogren_bas_col, sk_min_col, sk_max_col, yuksek_bit_col, sk_med_col, resolved_cols
)

ref_exp_series = df_pos[kid_poz_col] if (kid_poz_col and df_pos[kid_poz_col].notna().any()) else df_pos[kid_total_col] if (kid_total_col and df_pos[kid_total_col].notna().any()) else pd.Series([],dtype=float)
used_exp = cand_rel_exp if cand_rel_exp>0 else (cand_total_exp if cand_total_exp>0 else None)
q = exp_to_quantile(ref_exp_series, used_exp)
sal_arr = pd.to_numeric(df_pos[sal_col], errors="coerce").dropna().values
target, (blo,bhi) = suggest_range_from_scale(scmin, scmed, scmax, sal_arr, q, max_width=10_000)
band_text = f"{int(round(blo)):,} – {int(round(bhi)):,} TL".replace(",", ".")

# ===================== Header =====================
st.subheader("🎯 Pozisyona Göre Öneri (Skala İçinde)")
note = ""
if np.isfinite(scmax) and int(round(target)) >= int(round(scmax)):
    note = " (skala üst sınırına dayalı)"
st.markdown(f"""
<div class='card'>
<div><span class='eq-badge'>Önerilen hedef teklif</span> <b class='title-accent'>{int(round(target)):,} TL</b>{note}</div>
<div style='margin-top:6px'><span class='eq-badge'>Makul bant</span> <b>{band_text}</b></div>
<p class='small' style='margin-top:8px;'>Öneri ve bant <b>Öğrenen Başlangıç – Yüksek Bitiş</b> aralığını asla aşmaz. Skala yoksa iç dağılım p25/p50/p75 ile belirlenir. Bant genişliği ≤ 10.000 TL.</p>
</div>
""".replace(",", "."), unsafe_allow_html=True)

if proposed > 0 and np.isfinite(scmin) and np.isfinite(scmax) and scmin < scmax:
    if proposed > round(scmax):
        st.markdown("<div class='warn'>Girdiğin teklif <b>skala üstünde</b>.</div>", unsafe_allow_html=True)
    elif proposed < round(scmin):
        st.markdown("<div class='warn'>Girdiğin teklif <b>skala altında</b>.</div>", unsafe_allow_html=True)

def format_currency_display(value):
    if pd.isna(value) or not np.isfinite(value):
        return "0 TL"
    return f"{int(round(value)):,} TL".replace(",", ".")

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
with c1: st.metric("Skala Medyanı (dosya)", format_currency_display(scmed))
with c2: st.metric("Medyan (iç)", format_currency_display(p50))
with c3: st.metric("Ortalama (iç)", format_currency_display(np.nanmean(sal_arr) if sal_arr.size and np.isfinite(np.nanmean(sal_arr)) else 0))
with c4: st.metric("p25 (iç)", format_currency_display(p25))
with c5: st.metric("p75 (iç)", format_currency_display(p75))
with c6: st.metric("Min (iç)", format_currency_display(amin))
with c7: st.metric("Max (iç)", format_currency_display(amax))

# ===================== PR =====================
st.markdown("---")
st.subheader("📊 Piyasa PR Kıyas (yüzde)")
avg_pr = np.nan; pr_ratio_pct = np.nan
if pr_col and pr_col in df_pos.columns and df_pos[pr_col].notna().any():
    avg_pr = pd.to_numeric(df_pos[pr_col], errors="coerce").dropna().mean()
    st.metric("Ortalama PR (pozisyon)", f"%{avg_pr:,.2f}".replace(",", "."))
if proposed>0 and np.isfinite(avg_pr) and avg_pr>0:
    offer_scale = (proposed/1000.0) if proposed>=1000 else float(proposed)
    pr_ratio_pct = 100.0 * offer_scale / avg_pr
    st.metric("Teklif/Ortalama PR", f"%{pr_ratio_pct:,.2f}".replace(",", "."), help="%=100×Teklif(bin TL)/Ortalama PR(puan)")
else:
    st.info("PR verisi bu pozisyonda bulunamadı (örn. 5%, 0,05 veya 5).")

# ===================== Medyana göre fark =====================
st.markdown("---")
if proposed>0:
    base_med = scmed if np.isfinite(scmed) else p50
    diff_pct = np.nan if (not np.isfinite(base_med) or base_med==0) else (100.0*(proposed-base_med)/base_med)
    if np.isnan(diff_pct):
        st.info("Medyana göre fark hesaplanamadı.")
    else:
        arrow = "⬆️" if diff_pct>0 else "⬇️" if diff_pct<0 else "⟂"
        badge = "warn" if abs(diff_pct)>10 else "ok"
        st.markdown(f"<div class='{badge}'>Önerdiğin teklif: <b>{int(round(proposed)):,} TL</b> · Medyan farkı: <b>%{abs(diff_pct):,.1f}</b> {arrow}</div>".replace(",", "."), unsafe_allow_html=True)

# ===================== Bant Altı / Üstü (Cached) =====================
@st.cache_data(show_spinner=False, ttl=300)
def get_split_band_cached(processed_df_cache_key, pos_filter, include_sub_filter, manager_selection_list, sal_col, ogren_bas_col, sk_min_col, sk_max_col, yuksek_bit_col, band_konum, resolved_cols_dict):
    _, df_pos_for_cache = get_filtered_dfs(processed_df_cache_key, _df_processed, pos_filter, include_sub_filter, manager_selection_list, resolved_cols_dict)

    scmin_r = float(pd.to_numeric(df_pos_for_cache[ogren_bas_col], errors="coerce").dropna().median()) if ogren_bas_col and df_pos_for_cache[ogren_bas_col].notna().any() else (
        float(pd.to_numeric(df_pos_for_cache[sk_min_col], errors="coerce").dropna().median()) if sk_min_col and df_pos_for_cache[sk_min_col].notna().any() else np.nan
    )
    scmax_r = float(pd.to_numeric(df_pos_for_cache[yuksek_bit_col],errors="coerce").dropna().median()) if yuksek_bit_col and df_pos_for_cache[yuksek_bit_col].notna().any() else (
        float(pd.to_numeric(df_pos_for_cache[sk_max_col], errors="coerce").dropna().median()) if sk_max_col and df_pos_for_cache[sk_max_col].notna().any() else np.nan
    )
    under_df = pd.DataFrame(); over_df = pd.DataFrame()
    if np.isfinite(scmin_r) and np.isfinite(scmax_r) and scmin_r < scmax_r:
        sals = pd.to_numeric(df_pos_for_cache[sal_col], errors="coerce")
        under_df = df_pos_for_cache.loc[sals < scmin_r].copy()
        over_df = df_pos_for_cache.loc[sals > scmax_r].copy()
    else:
        if band_konum and band_konum in df_pos_for_cache.columns:
            ser = df_pos_for_cache[band_konum].astype(str).map(tr_fold)
            under_df = df_pos_for_cache[ ser.str.contains("alt", na=False) ].copy()
            over_df = df_pos_for_cache[ ser.str.contains("ust", na=False) | ser.str.contains("ustu", na=False) | ser.str.contains("ust u", na=False) ].copy()
    return under_df, over_df

under_df, over_df = get_split_band_cached(
    _df_raw_cached_key, pos, st.session_state["include_sub"], st.session_state["mgr_selection"],
    sal_col, ogren_bas_col, sk_min_col, sk_max_col, yuksek_bit_col, band_konum, resolved_cols
)

st.markdown("---")
st.subheader("📐 Skalaya Göre Sınır Dışı Çalışanlar (Bant Altı / Bant Üstü)")
colu, colo = st.columns(2)
with colu:
    st.metric("Bant Altı Sayı", f"{len(under_df)}")
with colo:
    st.metric("Bant Üstü Sayı", f"{len(over_df)}")

st.markdown("**Bant Altı**")
st.dataframe(build_display(under_df, sal_col, kid_total_col, hire_col, perf_col, artis_oran, refs=[c for c in _df_processed.columns if tr_fold(c)=="ref"], mgr_cols=[c for c in [yon1_col,yon2_col] if c]), use_container_width=True, height=200)
st.markdown("**Bant Üstü**")
st.dataframe(build_display(over_df, sal_col, kid_total_col, hire_col, perf_col, artis_oran, refs=[c for c in _df_processed.columns if tr_fold(c)=="ref"], mgr_cols=[c for c in [yon1_col,yon2_col] if c]), use_container_width=True, height=200)

# ===================== İç Eşitlik (herkes + kırmızı vurgulu) =====================
st.markdown("---")
st.subheader("⚖️ İç Eşitlik – Adaya Göre (Tüm Kişiler Listelenir)")

def equity_table_optimized(frame, title_suffix, sal_col, kid_total_col, kid_poz_col, hire_col, perf_col, yon1_col, yon2_col, proposed_salary, cand_total_experience, eq_threshold_pct):
    sen_col = kid_total_col or kid_poz_col
    if not sen_col or sen_col not in frame.columns:
        st.info(f"{title_suffix}: Kıdem kolonu bulunamadı. Lütfen Excel dosyanızda 'Kıdem' veya 'Harici+Dahili Kıdem' gibi bir sütun olduğundan emin olun."); return pd.DataFrame(), pd.DataFrame()

    cols_to_keep = [sen_col, sal_col]
    if hire_col: cols_to_keep.append(hire_col)
    if perf_col: cols_to_keep.append(perf_col)
    if yon1_col: cols_to_keep.append(yon1_col)
    if yon2_col: cols_to_keep.append(yon2_col)

    view = frame[cols_to_keep].copy()
    renames = {sen_col:"Kıdem (Yıl)", sal_col:"Ücret"}
    if hire_col: renames[hire_col] = "İşe Giriş"
    if perf_col: renames[perf_col] = "ŞUBAT 2025 PERFORMANS SONUÇLARI"
    if yon1_col: renames[yon1_col] = "1. Yönetici"
    if yon2_col: renames[yon2_col] = "2. Yönetici"
    view.rename(columns=renames, inplace=True)

    if "İşe Giriş" in view.columns:
        view["İşe Giriş"] = pd.to_datetime(view["İşe Giriş"], errors="coerce").dt.date

    if proposed_salary > 0:
        view_salaries = pd.to_numeric(view["Ücret"], errors="coerce")
        view["Ezilme %"] = 100.0 * (proposed_salary - view_salaries) / view_salaries
    else:
        view["Ezilme %"] = np.nan

    cand_exp = cand_total_experience if cand_total_experience > 0 else 0
    view["Kıdem≥Aday"] = (pd.to_numeric(view["Kıdem (Yıl)"], errors="coerce") >= cand_exp)

    view["Risk?"] = np.where(
        view["Kıdem≥Aday"] & (view["Ezilme %"] >= eq_threshold_pct) & view["Ezilme %"].notna(),
        "Evet",
        "Hayır"
    )

    def color_risk_cell(val):
        if val == "Evet":
            return 'background-color: #ffebee; color:#b71c1c'
        return ''

    fmt = {
        "Ücret": lambda x: "" if pd.isna(x) else f"{int(round(pd.to_numeric(x, errors='coerce'))):,}".replace(",", "."),
        "Ezilme %": lambda x: "" if pd.isna(x) else f"%{x:,.1f}".replace(",", ".")
    }

    st.markdown(f"**{title_suffix}** | Toplam: {len(view)} • Ezilme Riski (≥%{eq_threshold}): {(view['Risk?']=='Evet').sum()}")

    styled_df = view.style.applymap(color_risk_cell, subset=["Risk?"]).format(fmt)
    st.dataframe(styled_df, use_container_width=True, height=300) 

    risk_only = view[(view["Risk?"]=="Evet")].copy()
    if len(risk_only)>0:
        st.markdown("**(Kırmızı) Ezilenler – Detay**")
        st.dataframe(risk_only.style.format(fmt), use_container_width=True, height=200) 
    else:
        st.caption("Ezilme riski yok.")

    return view, risk_only

tab_all, tab_mgr = st.tabs(["Tüm Şirket (Pozisyona göre)", "Seçili Yönetici"])
with tab_all:
    eq_all, risk_all = equity_table_optimized(df_pos_all, "Tüm Şirket", sal_col, kid_total_col, kid_poz_col, hire_col, perf_col, yon1_col, yon2_col, proposed, cand_total_exp, eq_threshold)
with tab_mgr:
    if st.session_state["mgr_selection"]:
        eq_mgr, risk_mgr = equity_table_optimized(df_pos, "Seçili Yönetici", sal_col, kid_total_col, kid_poz_col, hire_col, perf_col, yon1_col, yon2_col, proposed, cand_total_exp, eq_threshold)
    else:
        st.info("Yönetici seçersen burada ayrıca filtreli tabloyu gösteririm.")
        eq_mgr, risk_mgr = pd.DataFrame(), pd.DataFrame()

# ===================== En Yakın İşe Giriş =====================
st.markdown("---")
st.subheader("🕒 Aynı Pozisyonda En Yakın İşe Girişler (Filtreli Veri)")

recent_view = pd.DataFrame()
if hire_col and hire_col in df_pos.columns:
    keep = [hire_col, sal_col]
    for c in [kid_total_col, perf_col, yon1_col, yon2_col]:
        if c and c in df_pos.columns: keep.append(c)
    r = df_pos[keep].copy()
    r[hire_col] = pd.to_datetime(r[hire_col], errors="coerce", cache=True)
    r = r.sort_values(hire_col, ascending=False).head(300)
    r[hire_col] = r[hire_col].dt.date
    renm = {hire_col:"İşe Giriş", sal_col:"Ücret"}
    if kid_total_col: renm[kid_total_col] = "Harici+Dahili Kıdem"
    if perf_col: renm[perf_col] = "ŞUBAT 2025 PERFORMANS SONUÇLARI"
    if yon1_col: renm[yon1_col] = "1. Yönetici"
    if yon2_col: renm[yon2_col] = "2. Yönetici"
    r.rename(columns=renm, inplace=True)
    recent_view = r.copy()
    st.dataframe(recent_view, use_container_width=True, height=300) 
else:
    st.info("İşe giriş tarihi kolonu bulunamadı.")

# ===================== Fark Yaratanlar =====================
st.markdown("---")
st.subheader("🌟 Fark Yaratanlar (Pozisyondaki Tüm Kişiler – Filtreli)")

fy_list_raw = pd.DataFrame(); fy_view = pd.DataFrame()
if perf_col and perf_col in df_pos.columns:
    mask_fy_all = df_pos[perf_col].astype(str).str.contains("fark", case=False, regex=False, na=False)
    keep_cols = [sal_col, perf_col]
    for c in [kid_total_col, hire_col, yon1_col, yon2_col]:
        if c and c in df_pos.columns: keep_cols.append(c)
    fy_list_raw = df_pos.loc[mask_fy_all, keep_cols].copy()
    if not fy_list_raw.empty:
        avg_fy = pd.to_numeric(fy_list_raw[sal_col], errors="coerce").mean()
        st.metric("Fark Yaratan Sayısı", f"{len(fy_list_raw)}")
        st.metric("Fark Yaratan Ortalama Ücret", format_currency_display(avg_fy))
        fy_view = build_display(fy_list_raw, sal_col, kid_total_col, hire_col, perf_col, refs=None, mgr_cols=[c for c in [yon1_col,yon2_col] if c])
        st.dataframe(fy_view, use_container_width=True, height=200) 
    else:
        st.info("Bu pozisyonda 'Fark Yaratan' yok.")
else:
    st.info("Performans kolonu bulunamadı.")

# ===================== Excel Rapor =====================
def write_excel_bytes():
    out = BytesIO()
    try:
        try: import xlsxwriter; engine = "xlsxwriter"
        except Exception: import openpyxl; engine = "openpyxl"
        with pd.ExcelWriter(out, engine=engine) as writer:
            pd.DataFrame({
                "Pozisyon":[pos],
                "Teklif [TL]":[proposed],
                "Önerilen Hedef [TL]":[int(round(target))],
                "Makul Bant":[f"{int(round(blo)):,} – {int(round(bhi)):,} TL".replace(",", ".")],
                "Skala (Öğrenen Baş.–Yüksek Bitiş)":[f"{'' if np.isnan(scmin) else int(round(scmin)):,} / {'' if np.isnan(scmax) else int(round(scmax)):,}".replace(",", ".")],
                "İç Medyan":[int(round(p50)) if np.isfinite(p50) else ""],
                "p25/p75 (iç)":[f"{int(round(p25)) if np.isfinite(p25) else ''}/{int(round(p75)) if np.isfinite(p75) else ''}"],
                "PR Ortalama [%]":[round(avg_pr,2) if np.isfinite(avg_pr) else ""],
                "Teklif/PR [%]":[round(pr_ratio_pct,2) if np.isfinite(pr_ratio_pct) else ""],
                "Deneyim q":[round(q,2)],
                "Yönetici filtresi":[", ".join(st.session_state["mgr_selection"]) if st.session_state["mgr_selection"] else "—"]
            }).to_excel(writer, index=False, sheet_name="Ozet")
            df_pos.to_excel(writer, index=False, sheet_name="PozisyonVeri_Filtreli")
            df_pos_all.to_excel(writer, index=False, sheet_name="PozisyonVeri_Tum")
            if len(under_df)>0:
                build_display(under_df, sal_col, kid_total_col, hire_col, perf_col, artis_oran, mgr_cols=[c for c in [yon1_col,yon2_col] if c]).to_excel(writer, index=False, sheet_name="BantAlti")
            if len(over_df)>0:
                build_display(over_df, sal_col, kid_total_col, hire_col, perf_col, artis_oran, mgr_cols=[c for c in [yon1_col,yon2_col] if c]).to_excel(writer, index=False, sheet_name="BantUstu")

            def _export(dfv, name):
                if isinstance(dfv, pd.io.formats.style.Styler):
                    dfv = dfv.data
                dfv.to_excel(writer, index=False, sheet_name=name)

            if len(eq_all)>0: _export(eq_all, "Icdeng_Tum")
            if len(risk_all)>0: _export(risk_all, "Icdeng_Tum_Risk")
            if st.session_state["mgr_selection"]:
                if len(eq_mgr)>0: _export(eq_mgr, "Icdeng_Yonetici")
                if len(risk_mgr)>0: _export(risk_mgr, "Icdeng_Yonetici_Risk")
            if not fy_view.empty: fy_view.to_excel(writer, index=False, sheet_name="FarkYaratanlar")
            if not recent_view.empty: recent_view.to_excel(writer, index=False, sheet_name="SonGirisler")
        out.seek(0); return out.getvalue()
    except Exception as e:
        st.error(f"Excel yazma hatası: {e}\nGerekirse: pip install xlsxwriter veya openpyxl"); return None

if st.button("📥 Raporu İndir (Excel)"):
    bytes_xlsx = write_excel_bytes()
    if bytes_xlsx:
        st.download_button("İndir (Excel)", data=bytes_xlsx, file_name=f"teklif_ozet_{pos.replace(' ','_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===================== E-posta (HTML + grafik inline + grafik ek + Excel ek) =====================
@st.cache_data(show_spinner=False, ttl=300)
def build_summary_chart_png_cached(vals_array, scmin, scmax, scmed, target_val, proposed_val, pos_title):
    fig, ax = plt.subplots(figsize=(6,3.2), dpi=150)
    if vals_array.size > 0:
        ax.hist(vals_array, bins=15)
    def mark(x, label):
        if np.isfinite(x):
            ax.axvline(x, linestyle="--"); ax.text(x, ax.get_ylim()[1]*0.95, label, rotation=90, va="top", ha="right", fontsize=8)
    mark(scmin,"Skala Min"); mark(scmax,"Skala Max"); mark(scmed,"Skala Medyan"); mark(int(round(target_val)),"Öneri"); mark(proposed_val if proposed_val>0 else np.nan,"Teklif")
    ax.set_xlabel("Ücret (TL)"); ax.set_ylabel("Adet"); ax.set_title(f"{pos_title} – Ücret Dağılımı (Filtreli)")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    png = buf.getvalue()
    temp_path=None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
            tf.write(png); temp_path=tf.name
    except: pass
    return png, temp_path

def df_to_html_table(df: pd.DataFrame, max_rows=10):
    if df is None or len(df)==0: return "<i>Veri yok</i>"
    view = df.copy()
    for c in view.columns:
        if "Ücret" in c or tr_fold(c)=="ucret":
            try:
                view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda v: f"{int(round(v)):,}".replace(",", ".") if pd.isfinite(v) else "")
            except: pass
    return view.head(max_rows).to_html(index=False, escape=False)

def build_email_html():
    rows = [
        ("Pozisyon", pos),
        ("Önerilen Hedef", f"{int(round(target)):,} TL".replace(",", ".")),
        ("Makul Bant", f"{int(round(blo)):,} – {int(round(bhi)):,} TL".replace(",", ".")),
        ("Skala (Öğrenen Baş.–Yüksek Bitiş)", f"{'' if np.isnan(scmin) else int(round(scmin)):,} / {'' if np.isnan(scmax) else int(round(scmax)):,}".replace(",", ".")),
        ("İç Medyan", f"{'' if np.isnan(p50) else int(round(p50)):,} TL".replace(",", ".")),
        ("Yönetici filtresi", ", ".join(st.session_state["mgr_selection"]) if st.session_state["mgr_selection"] else "—"),
        ("Ortalama PR", "" if np.isnan(avg_pr) else f"%{avg_pr:,.2f}".replace(",", ".")),
        ("Teklif/Ortalama PR", "" if np.isnan(pr_ratio_pct) else f"%{pr_ratio_pct:,.2f}".replace(",", ".")),
        ("Bant Altı / Üstü", f"{len(under_df)} / {len(over_df)}"),
    ]
    tbl = "".join([f"<tr><td style='padding:6px 10px;border:1px solid #e5e7eb;background:#fafafa'>{k}</td>"
                   f"<td style='padding:6px 10px;border:1px solid #e5e7eb'>{v}</td></tr>" for k,v in rows])

    under_html = df_to_html_table(build_display(under_df, sal_col, kid_total_col, hire_col, perf_col), 10)
    over_html = df_to_html_table(build_display(over_df, sal_col, kid_total_col, hire_col, perf_col), 10)
    eq_html = df_to_html_table(eq_mgr if st.session_state["mgr_selection"] else eq_all, 10)

    html = f"""
    <div style="font-family:Segoe UI,Arial,sans-serif;font-size:14px;color:#111">
    <h3 style="color:#1976D2;margin:0 0 10px">Ücret Teklif Özeti</h3>
    <table style="border-collapse:collapse">{tbl}</table>
    <p style="margin:10px 0 4px"><b>Ücret Dağılım Grafiği</b></p>
    <img src="cid:chart1" style="max-width:700px;border:1px solid #eee;border-radius:8px"/>
    <h4>Kanıt Tabloları</h4>
    <p><b>İç Eşitlik (hepsi)</b></p>{eq_html}
    <p><b>Bant Altı</b></p>{under_html}
    <p><b>Bant Üstü</b></p>{over_html}
    </div>
    """
    lines = [
        f"Pozisyon: {pos}",
        f"Önerilen hedef: {int(round(target)):,} TL".replace(",", "."),
        f"Makul bant: {int(round(blo)):,} – {int(round(bhi)):,} TL".replace(",", "."),
        f"Skala: {'' if np.isnan(scmin) else int(round(scmin)):,} / {'' if np.isnan(scmax) else int(round(scmax)):,}".replace(",", "."),
        f"İç Medyan: {'' if np.isnan(p50) else int(round(p50)):,} TL".replace(",", "."),
        f"Yönetici filtresi: {', '.join(st.session_state['mgr_selection']) if st.session_state['mgr_selection'] else '—'}"
    ]
    return "\n".join(lines), html

def send_email_smtp(host, port, user, password, to_addr, subject, html_body, text_body, attachment_bytes=None, attachment_name="rapor.xlsx", chart_png=None):
    msg = MIMEMultipart('related'); msg['From']=user; msg['To']=to_addr; msg['Subject']=subject
    alt = MIMEMultipart('alternative'); alt.attach(MIMEText(text_body, "plain","utf-8")); alt.attach(MIMEText(html_body,"html","utf-8")); msg.attach(alt)
    if chart_png:
        img = MIMEImage(chart_png, name="chart.png"); img.add_header('Content-ID','<chart1>'); img.add_header('Content-Disposition','inline', filename="chart.png"); msg.attach(img)
        partimg = MIMEBase("application","octet-stream"); partimg.set_payload(chart_png); encoders.encode_base64(partimg)
        partimg.add_header("Content-Disposition", 'attachment; filename="ucret_dagilimi.png"'); msg.attach(partimg)
    if attachment_bytes:
        part = MIMEBase("application","octet-stream"); part.set_payload(attachment_bytes); encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"'); msg.attach(part)
    with smtplib.SMTP(host, port) as server:
        server.starttls(); server.login(user, password); server.sendmail(user, [to_addr], msg.as_string())

def send_email_outlook(to_addr, subject, html_body, text_body, attach_path=None, chart_path=None):
    try:
        import win32com.client as win32
    except Exception as e:
        raise RuntimeError("Outlook için pywin32 gerekli: python -m pip install pywin32") from e
    outlook = win32.Dispatch("Outlook.Application"); mail = outlook.CreateItem(0)
    mail.To = to_addr; mail.Subject = subject; mail.HTMLBody = html_body or text_body
    if attach_path and os.path.exists(attach_path):
        mail.Attachments.Add(attach_path)
    if chart_path and os.path.exists(chart_path):
        att = mail.Attachments.Add(chart_path)
        PR_ATTACH_CONTENT_ID = "http://schemas.microsoft.com/mapi/proptag/0x3712001F"
        att.PropertyAccessor.SetProperty(PR_ATTACH_CONTENT_ID, "chart1")
    mail.Send()

text_body, html_body = build_email_html()

# ====================================================================================================
# DİKKAT: AŞAĞIDAKİ BİLGİLER DOĞRUDAN KODA GÖMÜLMÜŞTÜR. BU GÜVENLİK RİSKİ TAŞIR.
# YALNIZCA DENEME AMAÇLIDIR VE HASSAS BİLGİLERİNİZİ KİMSEYLE PAYLAŞMAYINIZ.
# MÜMKÜN OLAN EN KISA SÜREDE ORTAM DEĞİŞKENLERİNE GERİ DÖNMELİSİNİZ.
# ====================================================================================================

# Varsa kendi e-posta sunucunuzun bilgileriyle değiştirin. (Örn: Outlook için smtp.office365.com)
_EMBEDDED_SMTP_HOST = "smtp.gmail.com"
_EMBEDDED_SMTP_PORT = 587 # Gmail için 587 veya SSL için 465

# KENDİ E-POSTA ADRESİNİZİ YAZIN (gönderen)
_EMBEDDED_SMTP_USER = "sizin.eposta.adresiniz@gmail.com"

# GMAIL KULLANIYORSANIZ UYGULAMA ŞİFRENİZİ BURAYA YAZIN
# DİKKAT: Normal Gmail şifrenizi DEĞİL, yukarıdaki adımlarda oluşturduğunuz 16 haneli Uygulama Şifrenizi kullanın.
# Diğer e-posta sağlayıcıları için normal şifreniz olabilir.
_EMBEDDED_SMTP_PASS = "gmail_uygulama_sifreniz_buraya_yapistirin"

# E-POSTALARIN GÖNDERİLECEĞİ ALICI E-POSTA ADRESİNİ YAZIN
# Birden fazla kişi için virgülle ayırabilirsiniz (örn: "kisi1@mail.com,kisi2@mail.com")
_EMBEDDED_TO_EMAIL = "alicinin.eposta.adresi@domain.com"

_EMBEDDED_MAIL_SUBJECT = "Teklif Asistanı Raporu"
_EMBEDDED_ATTACH_REPORT = True
_EMBEDDED_MAIL_METHOD = "SMTP" # "SMTP" veya "Outlook (MAPI) – Windows"

# ====================================================================================================
# E-posta ayarları artık doğrudan yukarıdaki değişkenlerden alınacak.
# ====================================================================================================

try:
    sal_arr_for_plot = pd.to_numeric(df_pos[sal_col], errors="coerce").dropna().values
    chart_png, chart_path = build_summary_chart_png_cached(sal_arr_for_plot, scmin, scmax, scmed, target, proposed, pos)
except Exception as e:
    st.warning(f"Grafik oluşturulurken hata oluştu: {e}. E-postaya eklenemeyecek.")
    chart_png, chart_path = None, None

st.markdown("---")
st.subheader("✉️ E-posta")
st.caption("HTML gövde otomatik üretilir; Excel raporu ve grafik ek olarak eklenir (raporu önceden indirmen gerekmez).")

st.write("E-posta ayarları (Koda gömülü):")
st.code(f"""
MAIL_METHOD: {_EMBEDDED_MAIL_METHOD}
SMTP_HOST: {_EMBEDDED_SMTP_HOST}
SMTP_PORT: {_EMBEDDED_SMTP_PORT}
SMTP_USER: {_EMBEDDED_SMTP_USER}
TO_EMAIL: {_EMBEDDED_TO_EMAIL}
MAIL_SUBJECT: {_EMBEDDED_MAIL_SUBJECT}
ATTACH_REPORT: {_EMBEDDED_ATTACH_REPORT}
""")

st.warning("⚠️ DİKKAT: E-posta bilgileri doğrudan koda gömülüdür. Bu, güvenlik riski taşır. Hassas bilgilerinizi korumak için ortam değişkenlerini kullanmanız önerilir.")

show_html = st.checkbox("HTML önizleme", value=False)
st.text_area("Düz metin gövde", value=text_body, height=130)
if show_html:
    st.components.v1.html(html_body, height=420, scrolling=True)

if st.button("✉️ E-posta Gönder"):
    try:
        attach_bytes = None
        attach_path = None
        attach_name = f"teklif_ozet_{pos.replace(' ','_')}.xlsx"
        if _EMBEDDED_ATTACH_REPORT:
            bytes_xlsx = write_excel_bytes()
            if bytes_xlsx:
                attach_bytes = bytes_xlsx
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
                    tf.write(bytes_xlsx); attach_path = tf.name

        if _EMBEDDED_MAIL_METHOD == "Outlook (MAPI) – Windows":
            send_email_outlook(_EMBEDDED_TO_EMAIL, _EMBEDDED_MAIL_SUBJECT, html_body, text_body, attach_path if _EMBEDDED_ATTACH_REPORT else None, chart_path)
        elif _EMBEDDED_MAIL_METHOD == "SMTP":
            if not all([_EMBEDDED_SMTP_HOST, _EMBEDDED_SMTP_PORT, _EMBEDDED_SMTP_USER, _EMBEDDED_SMTP_PASS, _EMBEDDED_TO_EMAIL]):
                st.error("E-posta bilgileri eksik. Lütfen kod içindeki ayarları kontrol edin."); st.stop()
            send_email_smtp(_EMBEDDED_SMTP_HOST, int(_EMBEDDED_SMTP_PORT), _EMBEDDED_SMTP_USER, _EMBEDDED_SMTP_PASS, _EMBEDDED_TO_EMAIL, _EMBEDDED_MAIL_SUBJECT, html_body, text_body, attach_bytes if _EMBEDDED_ATTACH_REPORT else None, attach_name, chart_png)
        else:
            st.error(f"Geçersiz MAIL_METHOD: {_EMBEDDED_MAIL_METHOD}")
            
        st.success("E-posta gönderildi ✅ (Excel + grafik ekli).")
    except Exception as e:
        st.error(f"E-posta gönderilemedi: {e}")
    finally:
        if attach_path and os.path.exists(attach_path): os.remove(attach_path)
        if chart_path and os.path.exists(chart_path): os.remove(chart_path)

# ===================== Sohbet =====================
st.markdown("---")
st.subheader("💬 Sohbet (Beta)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def local_bot_answer(msg: str) -> str:
    m = msg.lower()
    why = ("neden" in m) or ("niye" in m) or ("gerekçe" in m) or ("sebep" in m)
    if why:
        parts = []
        parts.append(f"Öneri, aday deneyim kantiline göre (q≈{q:.2f}) hesaplandı.")
        if np.isfinite(scmin) and np.isfinite(scmax):
            parts.append(f"Skala aralığı {int(round(scmin)):,}–{int(round(scmax)):,} TL; hedef ve bant bu aralıktan çıkmaz.".replace(",", "."))
        else:
            parts.append(f"Skala yoksa iç dağılım p25/p50/p75 ({int(round(p25)) if np.isfinite(p25) else '-'} / {int(round(p50)) if np.isfinite(p50) else '-'} / {int(round(p75)) if np.isfinite(p75) else '-'}) baz alındı.".replace(",", "."))
        if proposed>0 and np.isfinite(scmed):
            parts.append(f"Girdiğin teklif medyana göre fark: %{(100*(proposed-scmed)/scmed):.1f}".replace(",", "."))
        return " ".join(parts)
    if "bant" in m: return "Bant altı/üstü: skala varsa min-max’a göre; yoksa Ücret Konumu ile sınıflandırıyorum."
    if "ezil" in m or "eşitlik" in m: return f"İç dengede tüm kişileri listelerim; kıdemi adaydan ≥ olan ve %{eq_threshold}+ ezilme yüzdesi çıkanları kırmızı vurgularım."
    if "pr" in m: return "PR oranı = 100 × teklif(bin TL) / ortalama PR."
    if "skala" in m: return "Skala: Öğrenen Başlangıç – Yüksek Bitiş; yoksa dosyadaki min/max ve iç dağılım."
    return "Detay iste: skala, iç dağılım, PR, bant altı/üstü, iç denge, yöneticiler…"

def openai_answer(msg: str) -> str:
    return local_bot_answer(msg)

if st.session_state.get("start_chat", False): # start_chat varsayılan olarak False
    colL, colM, colR = st.columns([1,2,1])
    with colM:
        st.markdown("<div class='chatbox'>", unsafe_allow_html=True)
        st.markdown("<div class='chatlog'>", unsafe_allow_html=True)
        for role, text in st.session_state.chat_history:
            css = "msgu" if role=="user" else "msga"
            st.markdown(f"<div class='{css}'><b>{'Sen' if role=='user' else 'Asistan'}:</b> {text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        user_msg = st.chat_input("Mesajını yaz ve Enter'a bas")
        if user_msg:
            st.session_state.chat_history.append(("user", user_msg))
            ans = openai_answer(user_msg) if st.session_state.get("chat_mode", "Yerel")=="OpenAI API (opsiyonel)" else local_bot_answer(user_msg)
            st.session_state.chat_history.append(("assistant", ans))
            st.rerun()
        if st.button("🧹 Sohbeti sıfırla"):
            st.session_state.chat_history = []; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Sohbeti başlatmak için soldan ‘Sohbeti başlat’ kutusunu işaretle.")
