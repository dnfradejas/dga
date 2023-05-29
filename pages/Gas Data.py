import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit_authenticator as stauth
import pickle
import os


#os.environ['OPENAI_API_KEY'] = 
st.set_page_config(layout='wide', 
                   page_title="Admin Access", 
                   initial_sidebar_state='expanded',
                   page_icon="📊",
                   )

st.sidebar.success("Select a page above")


# User Authentication
names = ["Miles Ramirez", "Edson David"]
usernames = ["mmramirez", "eddavid"]

# Load hashed passwords
file_path = Path(__file__).parent.parent / "hashed_pw.pkl"
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

    # set page rows and columns
    form_container = st.container()


    # read csv files
    df = pd.read_csv(Path(__file__).parent.parent / "all bank 1.csv")
    address = pd.read_csv(Path(__file__).parent.parent / "address.csv")

    
    def save_data(_df,data):
        """
        Save new data to csv
        """
        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([data], columns=_df.columns)
            
        # Append the new row to the existing DataFrame
        _df = _df.append(new_row_df, ignore_index=True)
        _df.to_csv(Path(__file__).parent.parent / "all bank 1.csv",index=False, mode='w')

    with form_container:
        data = [None]*10
        with st.form("add_gas_form"):
            st.subheader("Insert New Gas Readings")
            st.caption("Add new data to dashboard")

            station_options = address['SUBSTATION'].unique().tolist()      
                
            # Add a dropdown to select the station
            data[0] = st.selectbox("Please select substation",station_options)    

            # PXF
            data[1] = station_options.index(data[0]) + 1


            # add gas concentration inputs
            st.markdown('')
            col1, col2, col3, col4, col5 = st.columns(5)
              
            with col1:
                data[3] = st.number_input("Hydrogen Concentration", value=0.00)
            with col2:
                data[4] = st.number_input("Methane Concentration", value=0.00)   
            with col3:
                data[5] = st.number_input("Ethane Concentration", value=0.00)     
            with col4: 
                data[6] = st.number_input("Ethylene Concentration", value=0.00)     
            with col5:
                data[7] = st.number_input("Acetylene Concentration", value=0.00)     
                                      
               
            # Please select date
            st.markdown('')
            date = st.date_input("Please select date of reading", value=None)
            data[2] = date.strftime("%Y-%m-%d 00:00:00") if date else None
            
            # LAT
            data[8] = address.loc[address["SUBSTATION"] == data[0], "LAT"].values[0]

            # LNG
            data[9] = address.loc[address["SUBSTATION"] == data[0], "LNG"].values[0]

            # submit form
            st.markdown('')

            if st.form_submit_button("Submit"):
                if data[3] and data[4] and data[5] and data[6] and data[7]:
                    # save to csv the new data
                    save_data(df, data)

                    # prompt user that it is working
                    st.success("Data successfully added to dashboard")

                    # zero out the valeus of the array
                    data = []
                        
                    # # Rerun the Streamlit app
                    # st.experimental_rerun() 

                else:
                     st.error("Please fill in all the fields")  
