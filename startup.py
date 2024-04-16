import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data =pd.read_csv('startUp (1).csv')
# ADD a Title and a subheader
# st.title('START UP PROFIT PREDICTOR APP')
# st.subheader('Built By Salmon Crushers')
st.markdown("<h1 style = 'color: #F5EEE6; text-align: center; font-size: 60px; font-family: Verdana'>START UP PROFIT PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Salmon Crushers</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

#Add an Image
st.image('pngwing.com.png', width = 600, caption = 'Start Up Project')




#Add a project problem statement
st.header('Project Background Information',divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")  #Word Wrap in View

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("<p>By analyzing a diverse set of parameters, including Market Expense, Administrative Expense, and Research and Development Spending, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success</p>", unsafe_allow_html= True)


# To add a sidebar
# Sidebar Designs
st.sidebar.image('pngwing.com 2.png')

st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width= True)

st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.sidebar.markdown("<br>", unsafe_allow_html = True)

# User Inputs
rd_spend = st.sidebar.number_input('Research and Development Expense', data['R&D Spend'].min(),data['R&D Spend'].max())
admin = st.sidebar.number_input('Administrative Expense', data['Administration'].min(),data['Administration'].max())
mkt = st.sidebar.number_input('Marketing Expense', data['Marketing Spend'].min(),data['Marketing Spend'].max())
state = st.sidebar.selectbox('Company Location', data['State'].unique())

# Import Transformers
admin_scaler = joblib.load('Administration_scaler.pkl')
mkt_scaler = joblib.load('Marketing Spend_scaler.pkl')
rd_spend_scaler = joblib.load('R&D Spend_scaler.pkl')
state_encoder = joblib.load('state_encoder.pkl')


# User Input Dataframe
user_input = pd.DataFrame()
user_input['R&D Spend'] = [rd_spend ]
user_input['Administration'] = [admin]
user_input['Marketing Spend'] = [mkt]
user_input['State'] = [state]


st.markdown("<br>", unsafe_allow_html = True)
st.header('Input Variable')
st.dataframe(user_input, use_container_width = True)


# Transform users input according to training scale and encoding
user_input['R&D Spend'] = rd_spend_scaler.transform(user_input[['R&D Spend']])
user_input['Administration'] = admin_scaler.transform(user_input[['Administration']])
user_input['Marketing Spend'] = mkt_scaler.transform(user_input[['Marketing Spend']])
user_input['State'] = state_encoder.transform(user_input[['State']])

# st.header('Transformed Input Variable')
# st.dataframe(user_input, use_container_width = True)

# modelling ---
model = joblib.load('startUpModel.pkl')

if st.button('Predict Profitability'):
    predicted_profit = model.predict(user_input)
    st.success(f"Your Campany's Predicted Profit Is {predicted_profit[0].round(2)}")
    st.code