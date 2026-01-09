
import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Regressão FastAPI — Uma amostra", layout="centered")
st.title("Predição (uma amostra) — FastAPI + Streamlit")

st.markdown("""
Preencha os campos abaixo para **uma única amostra**. O app enviará listas com um elemento para cada campo, como o backend espera.
""")

# Helpers
def parse_bool_ui(val: bool):
    # Já vem como bool do Streamlit
    return bool(val)

def parse_float_text(s, allow_percent=False):
    if s is None or s == "":
        return None
    s2 = str(s).strip()
    try:
        if allow_percent and s2.endswith("%"):
            s2 = s2[:-1]
            return float(s2) / 100.0
        return float(s2)
    except:
        return None

def parse_int_text(s):
    if s is None or s == "":
        return None
    try:
        return int(float(str(s).strip()))
    except:
        return None

def parse_amenities(s):
    """
    Aceita:
    - String simples: "Wifi;Kitchen"
    - Formato CSV: "Wifi,Kitchen"
    - Formato do dataset: {"Wireless Internet","Air conditioning",Kitchen}
    Envia como string limpa (mantém compatível com vetorizadores), ou lista se você preferir.
    """
    if not s:
        return None
    s2 = s.strip()
    # Remove chaves e aspas do formato { ... }
    if s2.startswith("{") and s2.endswith("}"):
        inner = s2[1:-1]
        # divide por vírgula respeitando aspas
        parts = [p.strip().strip('"') for p in inner.split(",")]
        # junta com ponto e vírgula para manter como uma única string
        return ";".join([p for p in parts if p])
    # Se vier "Wifi;Kitchen" ou "Wifi,Kitchen", normaliza para ';'
    if "," in s2 and ";" not in s2:
        s2 = ";".join([p.strip() for p in s2.split(",")])
    return s2

with st.form("single_sample_form"):
    st.subheader("Campos de entrada")

    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.text_input("accommodates (int)", value="3")
        bathrooms = st.text_input("bathrooms (float)", value="1.0")
        latitude = st.text_input("latitude (float)", value="-25.43")
        longitude = st.text_input("longitude (float)", value="-49.27")
        number_of_reviews = st.text_input("number_of_reviews (int)", value="10")
        review_scores_rating = st.text_input("review_scores_rating (float)", value="95")
        bedrooms = st.text_input("bedrooms (int)", value="1")
        beds = st.text_input("beds (int)", value="1")
        host_response_rate = st.text_input("host_response_rate (float ou %)", value="90%")

    with col2:
        property_type = st.text_input("property_type (str)", value="Apartment")
        room_type = st.text_input("room_type (str)", value="Entire home/apt")
        amenities = st.text_input("amenities (str / {…} / sep por vírgula/;)", value='{"Wireless Internet","Air conditioning",Kitchen}')
        bed_type = st.text_input("bed_type (str)", value="Real Bed")
        cancellation_policy = st.text_input("cancellation_policy (str)", value="strict")
        cleaning_fee = st.checkbox("cleaning_fee (bool)", value=False)
        city = st.text_input("city (str)", value="Curitiba")
        host_has_profile_pic = st.checkbox("host_has_profile_pic (bool)", value=True)
        host_identity_verified = st.checkbox("host_identity_verified (bool)", value=True)
        instant_bookabl = st.checkbox("instant_bookabl (bool)", value=False)

    submit = st.form_submit_button("Prever")

if submit:
    # Converte tipos
    payload = {
        "accommodates": [parse_int_text(accommodates)],
        "bathrooms": [parse_float_text(bathrooms)],
        "latitude": [parse_float_text(latitude)],
        "longitude": [parse_float_text(longitude)],
        "number_of_reviews": [parse_int_text(number_of_reviews)],
        "review_scores_rating": [parse_float_text(review_scores_rating)],
        "bedrooms": [parse_int_text(bedrooms)],
        "beds": [parse_int_text(beds)],
        "property_type": [property_type or None],
        "room_type": [room_type or None],
        "amenities": [parse_amenities(amenities)],
        "bed_type": [bed_type or None],
        "cancellation_policy": [cancellation_policy or None],
        "cleaning_fee": [parse_bool_ui(cleaning_fee)],
        "city": [city or None],
        "host_has_profile_pic": [parse_bool_ui(host_has_profile_pic)],
        "host_identity_verified": [parse_bool_ui(host_identity_verified)],
        # Normaliza percentuais para 0–1
        "host_response_rate": [parse_float_text(host_response_rate, allow_percent=True)],
        "instant_bookabl": [parse_bool_ui(instant_bookabl)],
    }

    # Validação mínima
    missing_numeric = [
        k for k in ["accommodates", "bathrooms", "latitude", "longitude",
                    "number_of_reviews", "review_scores_rating", "bedrooms", "beds",
                    "host_response_rate"]
        if payload[k][0] is None
    ]
    if missing_numeric:
        st.error(f"Campos numéricos inválidos ou vazios: {', '.join(missing_numeric)}")
        st.stop()

    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            out = r.json()
            preds = out.get("predictions", [])
            st.success("Predição realizada com sucesso.")
            df_show = pd.DataFrame({k: [v[0]] for k, v in payload.items()})
            df_show["y_pred"] = preds if isinstance(preds, list) else [preds]
            st.dataframe(df_show, use_container_width=True)
        else:
            st.error(f"Falha na predição ({r.status_code}): {r.text}")
    except Exception as e:
        st.error(f"Erro ao contactar API: {e}")
