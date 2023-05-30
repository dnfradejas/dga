import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
from streamlit_folium import folium_static
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from style import Style 
import time
import multiprocessing as mp
from pathlib import Path
import streamlit_authenticator as stauth
import os
from datetime import datetime


#os.environ['OPENAI_API_KEY'] = 
st.set_page_config(layout='wide', 
                   page_title="Power Transformer DGA Monitoring", 
                   initial_sidebar_state='expanded',
                   page_icon="📊",
                   )

st.sidebar.success("Select a page above")

# User Authentication
names = ["Miles Ramirez", "Edson David"]
usernames = ["mmramirez", "eddavid"]

# Load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {
    "usernames":{
        usernames[0]:{
            "name":names[0],
            "password": hashed_passwords[0]
        },
        usernames[1]:{
            "name":names[1],
            "password": hashed_passwords[1]
        }
    }
}

authenticator = stauth.Authenticate(credentials,"Power Transformer DGA Monitoring","auth", cookie_expiry_days=30)

with st.sidebar:
    authenticator.logout("Logout", "sidebar")

name, authentication_status, username = authenticator.login("Login", "main")
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")



# if successful login:
if authentication_status:
    # re-arrange data
    df = pd.read_csv(Path(__file__).parent / "all bank 1.csv")
    df['DATE'] = pd.to_datetime(df['DATE'])
    address = pd.read_csv(Path(__file__).parent / "address.csv")
    model = pickle.load(open(Path(__file__).parent / 'model.pkl', 'rb'))
    latest = df.groupby('SUBSTATION').last()
    latest['SUBSTATION'] = latest.index
    latest = latest.reset_index(drop=True)
    X = latest.iloc[:,2:7]
    df2 = pd.DataFrame(model.predict_proba(X), columns=['Arc discharge', 'High overheating', 'Medium overheating', 'Normal',
        'Partial discharge', 'Spark discharge'])
    df2['max'] = df2.max(axis=1)
    df2['class'] = model.predict(X)




    # set page rows and columns
    
    container = st.container()
    col_normal, col_medium, col_high, col_spark, col_pd, col_arc = st.columns(6)
    st.markdown('')
    chart_col,checkbox_col= st.columns(2)
    addData_row = st.container()

    # setup style class
    style = Style() 

    #plot gas concentrations
    def plot(df, lat, lng):
        fig = make_subplots(shared_xaxes=True, 
            vertical_spacing=0.030, y_title = "Parts per Million")

        # Hydrogen Concentration
        fig.add_trace(go.Scatter(
            x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
            y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['HYDROGEN'],
            name='Hydrogen'
        ))

        # Methane Concentration
        fig.add_trace(go.Scatter(
            x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
            y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['METHANE'],
            name='Methane'
        ))

        # Ethane Concentration
        fig.add_trace(go.Scatter(
            x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
            y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHANE'],
            name='Ethane'
        ))

        # Ethlyene Concentration
        fig.add_trace(go.Scatter(
            x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
            y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ETHYLENE'],
            name='Ethylene'
        ))

        # Acetylene Concentration
        fig.add_trace(go.Scatter(
            x=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['DATE'],
            y=df[(df['LAT'] == lat) & (df['LNG'] == lng)]['ACETYLENE'],
            name='Acetylene'
        ))


        fig.update_layout(height=295, width=400,
                hovermode="x unified",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                yaxis=dict(dtick=150, title=""),

                )

        
        fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)', 'paper_bgcolor':'rgba(0,0,0,0)'})
        fig.update_xaxes(showticklabels=False)

    
        st.plotly_chart(fig, use_container_width=True)



    @st.cache_data(ttl=300)
    def get_current_time():
        """
        Refresh Page every 5 minutes
        """
        return time.strftime("%m/%d/%Y %I:%M %p")

    # calculate model confidence level
    def fault_score(df, address, lat, lng):
        """
        Update Fault Scores for High Overheating, Medium Overheating, Partial Discharge,
        Spark Discharge, and Arc Discharge
        """
        pxf = address[(address['LAT'] == lat) & (address['LNG'] == lng)]['SUBSTATION'].iloc[0]
        score = model.predict_proba(df[df['SUBSTATION'] == pxf].iloc[-1:None,3:8])
        score2 = model.predict_proba(df[df['SUBSTATION'] == pxf].iloc[-2:-1,3:8])
        delta = np.array(score)-np.array(score2)
        pxf_no = str(address[(address['LAT'] == lat) & (address['LNG'] == lng)].index[0]+1)
        
        
        st.markdown(style.style_metric(), unsafe_allow_html=True)

        with container:
            st.markdown("<h2>Real-time Dissolved Gas Analysis</h2>", unsafe_allow_html=True)        
            st.caption(f"As of {get_current_time()}")
        
        
        with col_medium:
            if round(delta[0][2]*100) == 0:       
                st.metric(label='Medium Overheating', value=str(round(score[0][2]*100))+"%")
            else:
                st.metric(label='Medium Overheating', value=str(round(score[0][2]*100))+"%", delta=str(round(delta[0][2]*100))+" %", delta_color='inverse')
        with col_high:
            if round(delta[0][1]*100) == 0:
                st.metric(label='High Overheating', value=str(round(score[0][1]*100))+"%")  
            else:
                st.metric(label='High Overheating', value=str(round(score[0][1]*100))+"%", delta=str(round(delta[0][1]*100))+" %", delta_color='inverse')
        with col_pd:
            if round(delta[0][4]*100) == 0:
                st.metric(label='Partial Discharge', value=str(round(score[0][4]*100))+"%")
            else:
                st.metric(label='Partial Discharge', value=str(round(score[0][4]*100))+"%", delta=str(round(delta[0][4]*100))+" %", delta_color='inverse')
        with col_spark:
            if round(delta[0][5]*100) == 0:
                st.metric(label='Spark Discharge', value=str(round(score[0][5]*100))+"%")
            else:
                st.metric(label='Spark Discharge', value=str(round(score[0][5]*100))+"%", delta=str(round(delta[0][5]*100))+" %", delta_color='inverse')
        with col_arc:
            if round(delta[0][0]*100) == 0:
                st.metric(label='Arc Discharge', value=str(round(score[0][0]*100))+"%")
            else:
                st.metric(label='Arc Discharge', value=str(round(score[0][0]*100))+"%", delta=str(round(delta[0][0]*100))+" %", delta_color='inverse')
        with col_normal:
            if round(delta[0][3]*100) == 0:
                st.metric(label='Normal', value=str(round(score[0][3]*100))+"%")
            else:
                st.metric(label='Normal', value=str(round(score[0][3]*100))+"%", delta=str(round(delta[0][3]*100))+" %", delta_color='inverse')


    def main():
    
        # will be used to select colors
        states = [':large_green_circle: Normal - Follow planned maintenance', 
                ':large_orange_circle: Caution - Schedule maintenance within 7 days.', 
                ':red_circle: Hazardous - Schedule maintenance within 48 hours.']
        colors = ['darkgreen', '#d45800', 'darkred']


        def get_color(index):
            """
            Return the color of the datapoints 
            """
            color = ""
            if (df2['class'][index] == 'Medium overheating') or (df2['class'][index] == 'High overheating'):
                color = '#d45800'
            elif (df2['class'][index] == 'Arc discharge') or (df2['class'][index] == 'Partial discharge') or (df2['class'][index] == 'Spark discharge'):
                color = 'darkred'
            else:
                color = 'darkgreen'    


            return color



        def filter_points(selected_states):
            """
            Function to filter and display the points on the map
            """
            m = folium.Map(location=[14.517855931485975, 121.06789220281651], zoom_start=7)
            folium.TileLayer('cartodbdark_matter').add_to(m)

            for i in range(len(address)):
                # get color of index
                color = get_color(i)

                # check if the value of state is selected
                if color in selected_states:
                    folium.CircleMarker(location=[address['LAT'][i], address['LNG'][i]], 
                                        radius=5,
                                        color=color,
                                        fill=True,
                                        fill_color=color,
                                        fill_opacity=1,
                                        tooltip=address['SUBSTATION'][i]).add_to(m)

            return m
        
        m = folium.Map(location=[14.517855931485975, 121.06789220281651], zoom_start=7)
        folium.TileLayer('cartodbdark_matter').add_to(m)

        with addData_row:
            st.subheader("Filter by Transforming States")
            st.caption("Show the different transformers in the map")
            selected_states = []
            for color, status in zip(colors, states):
                selected_option = st.checkbox(status, value=True, key=status)
                if selected_option:
                    selected_states.append(color)   
 
        with chart_col:
            st.subheader("Dissolved Gases Map")
            st.caption("Show the different transformers in the map")

            # Update the map
            updated_map = filter_points(selected_states)
            st_data = st_folium(updated_map,width=True, height=340)



        with checkbox_col:
            # Set title headers
            st.subheader("Dissolved Gas Concentrations")
            st.caption("Show the different gas concentration of the selected transformer.")

            # populate options
            station_options = address['SUBSTATION'].unique().tolist()      
            selected_station = st.selectbox("",station_options)    

            if st_data["last_object_clicked"] is None:
                lat = address[address['SUBSTATION'] == selected_station]['LAT'].iloc[0]
                lng = address[address['SUBSTATION'] == selected_station]['LNG'].iloc[0]
                plot(df, lat, lng)
                fault_score(df, address, lat, lng)
            
            elif st_data["last_object_clicked"] is not None:
                lat = st_data["last_object_clicked"]['lat']
                lng = st_data["last_object_clicked"]['lng']
                plot(df, lat, lng)
                fault_score(df, address, lat, lng)



    if __name__ == '__main__':
        main()
