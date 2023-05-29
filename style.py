import streamlit as st

class Style:
    """
    Class for UI Design
    """
    def style_metric(self):

        # Define the CSS style for the container header
        container_style = """
            <style>
                div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
                    overflow-wrap: break-word;
                    white-space: break-spaces;
                    color: white;

                }
                div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div p {
                    font-size: 90% !important;
                }

                div[data-testid="metric-container"] > div[data-testid="stMetricValue"] > div {
                    font-weight: bold !important;
                    color: orange !important;
                }

                div[data-baseweb="select"]{
                   width: 30%;

                }
                div[data-testid="stCaptionContainer"] > p {
                    color: white !important;
                }


                div[data-testid = "stHorizontalBlock"] > div[data-testid = "metric-container"] {
                   color: white !important;
                }

                div[data-testid="stCaptionContainer"] {
                    margin-top: -1em;
                }

                div[data-testid="stHorizontalBlock"] > div {
                    .css-k3w14i{
   
                    margin-bottom: 0 !important;
                    min-height: 0 !important;
                }

            </style>
        """

        return container_style