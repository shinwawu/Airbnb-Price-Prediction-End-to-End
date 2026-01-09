
import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Regressão FastAPI — Uma amostra", layout="centered")
st.title("Predição (uma amostra) — FastAPI + Streamlit")

st.markdown("""
Preencha os campos abaixo para **uma única amostra**.
O app enviará **apenas** as colunas esperadas pelo backend:
- Numéricas: `accommodates, bathrooms, latitude, longitude, number_of_reviews, review_scores_rating, bedrooms, beds`
- Categóricas (strings): `property_type, room_type, amenities, bed_type, cancellation_policy, cleaning_fee, city, host_has_profile_pic, host_identity_verified, host_response_rate, instant_bookable`
""")

# === Helpers ===
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

def parse_amenities(s: str):
    """
    Aceita:
    - "Wifi;Kitchen"
    - "Wifi,Kitchen"
    - {"Wireless Internet","Air conditioning",Kitchen}
    Normaliza para string única com ';' como separador.
    """
    if not s:
        return None
    s2 = s.strip()
    if s2.startswith("{") and s2.endswith("}"):
        inner = s2[1:-1]
        parts = [p.strip().strip('"') for p in inner.split(",")]
        return ";".join([p for p in parts if p])
    if "," in s2 and ";" not in s2:
        s2 = ";".join([p.strip() for p in s2.split(",")])
    return s2

def bool_to_tf_str(val: bool) -> str:
    """Converte bool para 't'/'f'."""
    return "t" if bool(val) else "f"

def bool_to_true_false_str(val: bool) -> str:
    """Converte bool para 'True'/'False' (string literal, como no dataset)."""
    return "True" if bool(val) else "False"

# === Form ===
with st.form("single_sample_form"):
    st.subheader("Campos de entrada")

    col1, col2 = st.columns(2)

    # Defaults inspirados em exemplo NYC/Manhattan
    with col1:
        accommodates = st.text_input("accommodates (int)", value="4")
        bathrooms = st.text_input("bathrooms (float)", value="1.0")
        latitude = st.text_input("latitude (float)", value="40.754321")
        longitude = st.text_input("longitude (float)", value="-73.983210")
        number_of_reviews = st.text_input("number_of_reviews (int)", value="15")
        review_scores_rating = st.text_input("review_scores_rating (float)", value="96")
        bedrooms = st.text_input("bedrooms (int)", value="2")
        beds = st.text_input("beds (int)", value="2")

    with col2:
        property_type = st.selectbox("property_type (str)", ["Apartment", "House", "Condominium", "Loft", "Other"], index=0)
        room_type = st.selectbox("room_type (str)", ["Entire home/apt", "Private room", "Shared room", "Hotel room"], index=0)
        amenities = st.text_input(
            "amenities",
            value='{"Wireless Internet","Air conditioning",Kitchen,Heating,"Family/kid friendly",Essentials,"Hair dryer",Iron,"Smoke detector","Fire extinguisher"}'
        )

        bed_type = st.selectbox("bed_type (str)", ["Real Bed", "Pull-out Sofa", "Futon", "Airbed", "Couch"], index=0)
        cancellation_policy = st.selectbox("cancellation_policy (str)", ["flexible", "moderate", "strict", "super_strict_30"], index=0)
        cleaning_fee_bool = st.checkbox("cleaning_fee (bool → será enviado como string 'True'/'False')", value=True)
        city = st.text_input("city (str)", value="NYC")
        host_has_profile_pic_bool = st.checkbox("host_has_profile_pic (bool → será enviado como 't'/'f')", value=True)
        host_identity_verified_bool = st.checkbox("host_identity_verified (bool → será enviado como 't'/'f')", value=True)
        host_response_rate_raw = st.text_input("host_response_rate (string: ex. '95%' ou '0.95')", value="95%")
        instant_bookable_bool = st.checkbox("instant_bookable (bool → será enviado como 't'/'f')", value=True)

    submit = st.form_submit_button("Prever")

if submit:
    # === Conversões numéricas ===
    accommodates_v = parse_int_text(accommodates)
    bathrooms_v = parse_float_text(bathrooms)
    latitude_v = parse_float_text(latitude)
    longitude_v = parse_float_text(longitude)
    number_of_reviews_v = parse_int_text(number_of_reviews)
    review_scores_rating_v = parse_float_text(review_scores_rating)
    bedrooms_v = parse_int_text(bedrooms)
    beds_v = parse_int_text(beds)

    # Validação mínima numérica
    missing_numeric = [
        k for k, v in {
            "accommodates": accommodates_v,
            "bathrooms": bathrooms_v,
            "latitude": latitude_v,
            "longitude": longitude_v,
            "number_of_reviews": number_of_reviews_v,
            "review_scores_rating": review_scores_rating_v,
            "bedrooms": bedrooms_v,
            "beds": beds_v,
        }.items() if v is None
    ]
    if missing_numeric:
        st.error(f"Campos numéricos inválidos ou vazios: {', '.join(missing_numeric)}")
        st.stop()

    # === Monta payload APENAS com num_cols + cat_cols ===
    payload = {
        # num_cols
        "accommodates": [accommodates_v],
        "bathrooms": [bathrooms_v],
        "latitude": [latitude_v],
        "longitude": [longitude_v],
        "number_of_reviews": [number_of_reviews_v],
        "review_scores_rating": [review_scores_rating_v],
        "bedrooms": [bedrooms_v],
        "beds": [beds_v],
        # cat_cols (strings)
        "property_type": [property_type or None],
        "room_type": [room_type or None],
        "amenities": [parse_amenities(amenities)],
        "bed_type": [bed_type or None],
        "cancellation_policy": [cancellation_policy or None],
        "cleaning_fee": [bool_to_true_false_str(cleaning_fee_bool)],  # "True"/"False"
        "city": [city or None],
        "host_has_profile_pic": [bool_to_tf_str(host_has_profile_pic_bool)],  # "t"/"f"
        "host_identity_verified": [bool_to_tf_str(host_identity_verified_bool)],  # "t"/"f"
        "host_response_rate": [host_response_rate_raw.strip() or None],  # string
        "instant_bookable": [bool_to_tf_str(instant_bookable_bool)],  # "t"/"f"
    }

    # === Chamada ===
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)

        st.markdown("### Debug do Payload Enviado")
        st.code(pd.DataFrame(payload).to_string(index=False), language="text")

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

