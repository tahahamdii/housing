import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:5000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #28a745;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if the Flask API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except requests.exceptions.RequestException:
        return False, None

def get_api_info():
    """Get API model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def make_prediction(data, endpoint="simple"):
    """Make prediction via API"""
    try:
        url = f"{API_BASE_URL}/predict/{endpoint}" if endpoint == "simple" else f"{API_BASE_URL}/predict"
        
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json() if response.text else {"error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return False, {"error": f"Connection error: {str(e)}"}

def create_price_chart(prediction_data):
    """Create price visualization"""
    if not prediction_data or not prediction_data.get('success'):
        return None
    
    price = prediction_data['predicted_price']
    area = prediction_data.get('input_summary', {}).get('area', 1000)
    price_per_sqft = price / area
    
    # Create comparison data
    comparison_data = {
        'Metric': ['Total Price', 'Price per Sq Ft', 'Market Average', 'Premium Range'],
        'Value': [price, price_per_sqft, price * 0.9, price * 1.2],
        'Type': ['Your Property', 'Per Sq Ft', 'Market Ref', 'Premium Ref']
    }
    
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        df, 
        x='Metric', 
        y='Value',
        color='Type',
        title="Price Analysis",
        color_discrete_map={
            'Your Property': '#28a745',
            'Per Sq Ft': '#17a2b8',
            'Market Ref': '#6c757d',
            'Premium Ref': '#fd7e14'
        }
    )
    
    fig.update_layout(
        showlegend=True,
        xaxis_title="",
        yaxis_title="Value ($)",
        title_x=0.5,
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("<h1 class='main-header'>🏠 Housing Price Predictor</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    Get instant house price predictions using our AI-powered API
    </p>
    """, unsafe_allow_html=True)
    
    # Check API status
    api_online, health_data = check_api_status()
    
    if api_online:
        st.markdown("""
        <div class='api-status status-online'>
        ✅ API Status: Online and Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='api-status status-offline'>
        ❌ API Status: Offline - Please start the Flask API server
        </div>
        """, unsafe_allow_html=True)
        st.error("🚨 Flask API is not running. Please start it with: `python flask_api.py`")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🏠 Property Details")
        
        # Basic property information
        st.subheader("Basic Information")
        area = st.number_input(
            "🏠 Area (sq ft)",
            min_value=500,
            max_value=20000,
            value=2000,
            step=100,
            help="Total area of the property in square feet"
        )
        
        bedrooms = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        bathrooms = st.number_input(
            "🚿 Bathrooms", 
            min_value=1,
            max_value=8,
            value=2,
            step=1
        )
        
        stories = st.number_input(
            "🏢 Stories",
            min_value=1,
            max_value=4,
            value=1,
            step=1
        )
        
        parking = st.number_input(
            "🚗 Parking Spaces",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )
        
        # Property features
        st.subheader("Property Features")
        mainroad = st.selectbox("🛣️ Main Road Access", ["yes", "no"])
        guestroom = st.selectbox("🏠 Guest Room", ["no", "yes"])
        basement = st.selectbox("🏠 Basement", ["no", "yes"])
        hotwaterheating = st.selectbox("♨️ Hot Water Heating", ["no", "yes"])
        airconditioning = st.selectbox("❄️ Air Conditioning", ["no", "yes"])
        prefarea = st.selectbox("⭐ Preferred Area", ["no", "yes"])
        furnishingstatus = st.selectbox(
            "🛋️ Furnishing Status", 
            ["unfurnished", "semi-furnished", "furnished"]
        )
        
        # Prediction button
        predict_btn = st.button("🔮 Get Price Prediction", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_btn:
            # Prepare data for API
            property_data = {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "mainroad": mainroad,
                "guestroom": guestroom,
                "basement": basement,
                "hotwaterheating": hotwaterheating,
                "airconditioning": airconditioning,
                "parking": parking,
                "prefarea": prefarea,
                "furnishingstatus": furnishingstatus
            }
            
            with st.spinner("🤖 Getting prediction from AI model..."):
                success, result = make_prediction(property_data, "simple")
                
                if success:
                    # Display main result
                    price = result['predicted_price']
                    st.markdown(
                        f"<div class='price-display'>💰 ${price:,.2f}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Display metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        price_per_sqft = result.get('price_per_sqft', price/area)
                        st.metric(
                            "Price per Sq Ft",
                            f"${price_per_sqft:.2f}",
                            delta=None
                        )
                    
                    with col_b:
                        price_per_room = price / (bedrooms + bathrooms)
                        st.metric(
                            "Price per Room",
                            f"${price_per_room:,.0f}",
                            delta=None
                        )
                    
                    with col_c:
                        st.metric(
                            "Prediction Time",
                            "< 1 sec",
                            delta=None
                        )
                    
                    # Show visualization
                    chart = create_price_chart(result)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Detailed breakdown
                    with st.expander("📊 Detailed Breakdown"):
                        st.json(result)
                        
                        # Property summary
                        st.write("**Property Summary:**")
                        summary = result.get('input_summary', {})
                        st.write(f"• Total Area: {summary.get('area', area):,} sq ft")
                        st.write(f"• Bedrooms: {summary.get('bedrooms', bedrooms)}")
                        st.write(f"• Bathrooms: {summary.get('bathrooms', bathrooms)}")
                        st.write(f"• Location Quality: {summary.get('location_quality', 'Standard')}")
                else:
                    st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        # Info panel
        st.markdown("### 📊 API Information")
        
        if api_online:
            api_info = get_api_info()
            if api_info:
                st.success(f"🤖 Model: {api_info.get('model_type', 'Unknown')}")
                st.info(f"📊 Features: {api_info.get('num_features', 0)}")
                
                with st.expander("🔍 Model Details"):
                    st.json(api_info)
        
        # Usage tips
        st.markdown("### 💡 Tips")
        st.info("""
        **For better predictions:**
        
        🏠 **Area**: Larger properties typically cost more
        
        🛏️ **Rooms**: More bedrooms = higher value
        
        ⭐ **Features**: Premium features like AC, basement increase value
        
        🛣️ **Location**: Main road access affects pricing
        """)
        
        # Recent predictions (mock data)
        with st.expander("📈 Recent Predictions"):
            recent_data = [
                {"Area": "1,800 sq ft", "Bedrooms": 3, "Price": "$245,000"},
                {"Area": "2,500 sq ft", "Bedrooms": 4, "Price": "$320,000"},
                {"Area": "1,200 sq ft", "Bedrooms": 2, "Price": "$180,000"},
            ]
            for pred in recent_data:
                st.write(f"• {pred['Area']}, {pred['Bedrooms']}BR → {pred['Price']}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    🏠 Housing Price Predictor • API-Powered • Built with Streamlit<br>
    <small>Make sure Flask API is running on localhost:5000</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()