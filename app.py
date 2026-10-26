import streamlit as st
import numpy as np
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Seattle House Valuation & Classification",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-expensive {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .status-affordable {
        color: #16A34A;
        font-weight: 700;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model assets
@st.cache_data
def load_assets():
    with open('model_assets.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Load assets
try:
    assets = load_assets()
    feature_cols = assets['feature_cols']
    cities = assets['cities']
    median_price = assets['median_price']
    mu = np.array(assets['mu'])
    sigma = np.array(assets['sigma'])
    reg_w = np.array(assets['reg_weights'])
    reg_b = assets['reg_bias']
    clf_w = np.array(assets['clf_weights'])
    clf_b = assets['clf_bias']
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">🏡 Seattle House Price Prediction & Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict exact market price (Linear Regression v2) & classify affordability (Logistic Regression v3) using machine learning models implemented from scratch with NumPy.</div>', unsafe_allow_html=True)

# Layout: Sidebar for Property Controls, Main area for Output
with st.sidebar:
    st.header("⚙️ Property Specifications")
    
    city = st.selectbox("📍 Location / City", cities, index=cities.index("Seattle") if "Seattle" in cities else 0)
    
    st.subheader("🏠 Size & Layout")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        bedrooms = st.number_input("Bedrooms", min_value=1.0, max_value=9.0, value=3.0, step=1.0)
        floors = st.number_input("Floors", min_value=1.0, max_value=3.5, value=1.5, step=0.5)
    with col_sb2:
        bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.25)
        condition = st.slider("Condition (1 to 5)", min_value=1, max_value=5, value=3)

    sqft_living = st.slider("Living Area (sqft)", min_value=400, max_value=10000, value=1900, step=50)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=6500, step=250)
    sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=300, step=50)
    
    st.subheader("🔨 Age & Renovation")
    yr_built = st.slider("Year Built", min_value=1900, max_value=2014, value=1975)
    has_renovated = st.checkbox("Has been renovated?", value=False)
    yr_renovated = yr_built if not has_renovated else st.slider("Year Renovated", min_value=yr_built, max_value=2014, value=max(yr_built, 2000))
    
    st.subheader("🌊 Premium Features")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        waterfront = 1 if st.checkbox("Waterfront property", value=False) else 0
    with col_p2:
        view = st.selectbox("View Rating (0-4)", [0, 1, 2, 3, 4], index=0)

# Feature Construction
renovated = 1 if (has_renovated and yr_renovated > 0) else 0
age = 2014 - yr_built

input_dict = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'sqft_basement': sqft_basement,
    'renovated': renovated,
    'age': age
}

# Construct the exact 55-feature vector
feature_vector = np.zeros(len(feature_cols))

for idx, col_name in enumerate(feature_cols):
    if col_name in input_dict:
        feature_vector[idx] = input_dict[col_name]
    elif col_name.startswith('city_'):
        feature_city = col_name.replace('city_', '')
        feature_vector[idx] = 1.0 if feature_city == city else 0.0

# Apply Training Z-Score Normalization
feature_vector_norm = (feature_vector - mu) / sigma

# Model Predictions
# 1. Linear Regression v2 (Dollar Price)
pred_log_price = np.dot(feature_vector_norm, reg_w) + reg_b
pred_dollar_price = np.exp(pred_log_price)

# 2. Logistic Regression v3 (Affordability Probability)
z_clf = np.dot(feature_vector_norm, clf_w) + clf_b
prob_expensive = sigmoid(z_clf)
is_expensive = prob_expensive >= 0.5

# Display Results
col1, col2, col3 = st.columns([1.3, 1.3, 1.4])

with col1:
    st.markdown("### 💰 Estimated Valuation")
    st.markdown(f"<h1 style='color: #2563EB; font-size: 2.8rem;'>${pred_dollar_price:,.0f}</h1>", unsafe_allow_html=True)
    st.caption("Multiple Linear Regression (v2 Model, R² = 0.722, 55 features)")
    diff_from_median = pred_dollar_price - median_price
    if diff_from_median >= 0:
        st.markdown(f"📈 **+${diff_from_median:,.0f}** above Seattle Median (${median_price:,.0f})")
    else:
        st.markdown(f"📉 **-${abs(diff_from_median):,.0f}** below Seattle Median (${median_price:,.0f})")

with col2:
    st.markdown("### 🏷️ Category Status")
    if is_expensive:
        st.markdown("<div class='status-expensive'>💎 EXPENSIVE CLASS (y = 1)</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{prob_expensive * 100:.1f}%` probability")
    else:
        st.markdown("<div class='status-affordable'>✅ AFFORDABLE CLASS (y = 0)</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{(1 - prob_expensive) * 100:.1f}%` probability")
    st.caption("Logistic Regression (v3 Model, 85.32% Test Accuracy)")

with col3:
    st.markdown("### 📊 Market Context")
    st.progress(float(min(max(prob_expensive, 0.0), 1.0)))
    st.markdown(f"""
    - **Seattle Median:** `${median_price:,.0f}`
    - **Price / Sqft:** `${pred_dollar_price / sqft_living:,.1f} / sqft`
    - **Property Age:** `{age} years old`
    """)

st.divider()

# Model Explanations & Comparative Architecture
st.subheader("📈 Architectural Insights & From-Scratch Implementation")

tab1, tab2, tab3 = st.tabs(["Regression Analysis (v1 & v2)", "Classification (v3)", "Key Influencing Factors"])

with tab1:
    st.markdown("""
    - **Pipeline Evolution:** v1 used 8 raw numeric features with loop-based gradient descent ($R^2 = 0.491$).
    - **v2 Enhancements:** Trimmed top 0.5% outliers ($>\\$675.13/\\text{sqft}$), dropped collinear `sqft_above`, added house age & renovation signals, and one-hot encoded 44 cities.
    - **Performance Gain:** RMSE dropped by **25.6%** to **0.2721**, and $R^2$ jumped to **0.7217** (72.2% variance explained).
    - **Speedup:** Pure vectorized NumPy operations accelerated execution by **1000×** (~0.07 ms per iteration).
    """)

with tab2:
    st.markdown(f"""
    - **Binary Objective:** Formulated around Seattle's median price (`${median_price:,.0f}`) to create an exactly balanced 50/50 dataset.
    - **High Accuracy:** Test Accuracy reached **85.32%**, significantly outperforming the dumb majority baseline (**49.34%**).
    - **Regularization Behavior:** Demonstrated that under large sample sizes ($m=3,620$), $L_2$ penalty is stable, while in data-scarce regimes ($m=50 < n=55$), it prevents runaway weight explosion.
    """)

with tab3:
    st.markdown(f"""
    **Current Property Dominant Factors in {city}:**
    - `sqft_living` (+2.44 weight): Dominant driver of high valuation and expensive classification.
    - `waterfront` (+0.38 weight): Huge premium factor if present.
    - `city_{city}`: Geographical neighborhood contribution specific to King County/Seattle area.
    """)
