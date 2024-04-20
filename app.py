# EDA Packages
import pandas as pd
import numpy as np
import joblib as jb
import plotly.express as px
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio # for new templates
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

# Modeling Packagaes
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
import pickle
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , RobustScaler , PolynomialFeatures
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.compose import ColumnTransformer

# Deployment Package
import streamlit as st
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# Raad csv for deployment & Set Page configration
df_new = pd.read_csv("new_df.csv").drop('Unnamed: 0',axis=1).sort_values(by=['date' , 'store'], ascending=True).reset_index(drop=True)
st.set_page_config(page_title='Walmart Predection Project' , layout = 'wide' , page_icon = '📊')
st.title('📊 Walmart Sales Prediction Project 🎯')

Brief = st.sidebar.checkbox(":blue[Brief About Data]")
Planning = st.sidebar.checkbox(":orange[About Project]")
About_me = st.sidebar.checkbox(":green[About Me]")
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# Brief
if Brief:
    st.sidebar.markdown(":blue[Brief About Data]")
    st.sidebar.write("""
    * One of the leading retail stores in the US, Walmart, would like to predict the sales and demand accurately. 
    * There are certain events and holidays which impact sales on each day. There are sales data available for 45 stores of Walmart. 
    * The business is facing a challenge due to unforeseen demands and runs out of stock some times, due to the inappropriate machine learning algorithm. 
    
    * An ideal ML algorithm will predict demand accurately and ingest factors like economic conditions including CPI, Unemployment Index, etc.
    * :red[So let us see the insights 👀.]
    """)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# About Project
if Planning:
    st.sidebar.markdown(":orange[About Project]")
    st.sidebar.write("""
    * Project Source: https://www.kaggle.com/datasets/yasserh/walmart-dataset
    * This is a project during Data Science Bootcamp @ZeroGrad under Mentoring of : Eng. Ahmed Mostafa
    * ZeroGrad:
        * Website: https://zero-grad.com/
        * Linkedin: https://www.linkedin.com/company/zero-grad
        * Youtube: https://www.youtube.com/c/zerograd
    """)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
if About_me :
    st.sidebar.markdown(":green[About me]")
    st.sidebar.write("""
    - Osama SAAD
    - Infor Master Data Management and Assets Control Section Head
        - Ibnsina Pharma
    - LinkedIn: 
        https://www.linkedin.com/in/osama-saad-samnudi/
    - Github : 
        https://github.com/OsamaSamnudi
    """)
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Tabs
Exploration , Insights , Dashboard , Prediction = st.tabs(['🔬 Exploration' , '💡 Insights' , '📊 Dashboard' , '🎯 Prediction'])
with Exploration:
    st.header('Exploration')
    with st.container():
        st.success('Sample of Data')
        st.dataframe(data= df_new.sample(10) , use_container_width=True)
        Data_00 , Data_01 , Data_02 = st.columns([70,3,55])
        with Data_00:
            st.success('Describe Numerical')
            st.dataframe(data= df_new.describe(include = 'number').replace(np.nan , '') , use_container_width=True)
        with Data_02:
            st.success('Describe Categorical')
            st.dataframe(data= df_new.describe(exclude = 'number') , use_container_width=True)
        
        Expl_00 , Expl_01 , Expl_02 = st.columns([50,50,50])
        def Cross_Tab(data , idx , col , val , func):
            return pd.crosstab(index=data[idx],columns=data[col] , values=data[val] , aggfunc=func).replace(np.nan , 0).style.background_gradient(cmap='twilight_shifted')
        with Expl_00:
            st.success('Avg Sales per Store in from Store Number 1 to 15')
            # Cross_Tab Season vs Year
            Plot_1 = Cross_Tab(df_new[df_new['store'].isin(range(1,16))] ,'store' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_1, use_container_width=True , height=600)

        with Expl_01:
            st.success('Avg Sales per Store in from Store Number 16 to 30')
            # Cross_Tab Season vs Year
            Plot_2 = Cross_Tab(df_new[df_new['store'].isin(range(16,31))] ,'store' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_2, use_container_width=True , height=600)

        with Expl_02:
            st.success('Avg Sales per Store in from Store Number 31 to 44')
            # Cross_Tab Season vs Year
            Plot_3 = Cross_Tab(df_new[df_new['store'].isin(range(31,45))] ,'store' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_3, use_container_width=True, height=600)

    with st.container():
        Expl_03 , Expl_04 , Expl_05 , Expl_06 = st.columns([75,60,65,60])
        with Expl_03:
            st.success('Avg Sales per month based on year')
            # Cross_Tab Season vs Year
            Plot_4 = Cross_Tab(df_new ,'month' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_4, use_container_width=True, height=500)

        with Expl_04:
            st.success('Avg Sales per season based on year')
            # Cross_Tab Season vs Year
            Plot_5 = Cross_Tab(df_new , 'season' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_5, use_container_width=True, height=200)

        with Expl_05:
            st.success('Avg Sales per temperature_class based on year')
            # Cross_Tab Season vs Year
            Plot_6 = Cross_Tab(df_new , 'temperature_class' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_6, use_container_width=True, height=260)

        with Expl_06:
            st.success('Avg Sales per holiday_flag based on year')
            # Cross_Tab Season vs Year
            Plot_7 = Cross_Tab(df_new , 'holiday_flag' , 'year' , 'weekly_sales' , 'mean')
            st.dataframe(data= Plot_7, use_container_width=True, height=150)

    with st.container():
        Expl_07 , Expl_08 , Expl_09 = st.columns([60,60,60])
        Fig_1 = px.histogram(df_new , x = 'temperature', text_auto=True , marginal='box' , title = "dist of temperature" , color_discrete_sequence= px.colors.qualitative.Dark24_r)
        Fig_2 = px.histogram(df_new , x = 'fuel_price', text_auto=True , marginal='box', title = "dist of fuel_price" , color_discrete_sequence=['darkseagreen'])
        Fig_3 = px.histogram(df_new , x = 'cpi', text_auto=True , marginal='box', title = "dist of cpi" , color_discrete_sequence= px.colors.qualitative.Bold)
        with Expl_07:
            st.plotly_chart(Fig_1 , use_container_width=True)
        with Expl_08:
            st.plotly_chart(Fig_2 , use_container_width=True)
        with Expl_09:
            st.plotly_chart(Fig_3 , use_container_width=True)
            
    with st.container():
        Expl_10 , Expl_11 , Expl_12 = st.columns([60,10,60])
        Fig_4 = px.histogram(df_new , x = 'unemployment', text_auto=True , marginal='box', title = "dist of unemployment" , color_discrete_sequence= px.colors.qualitative.G10)
        Fig_5 = px.histogram(df_new , x = 'weekly_sales', text_auto=True , marginal='box', title = "dist of weekly_sales" , color_discrete_sequence= px.colors.qualitative.Antique)
        with Expl_10:
            st.plotly_chart(Fig_4 , use_container_width=True)
        with Expl_12:
            st.plotly_chart(Fig_5 , use_container_width=True)

    with st.container():
            corr = df_new.select_dtypes(include='number').corr()
            IMSHOW = px.imshow(corr , width=1200,height=1200,text_auto=True,title='Correlation' , color_continuous_scale='viridis')
            IMSHOW.update_layout(annotations=[dict(font_size=90)])
            st.plotly_chart(IMSHOW , use_container_width=True)
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

with Insights:
    st.header('Insights')
    st.warning("""
    - Feature Correlation Analysis:
        - This section aims to uncover correlations between various features and the target variable "Weekly Sales," providing valuable insights for decision-makers.
        - We will visually explore patterns observed during our Exploratory Data Analysis (EDA) of sales across different time periods, encompassing year, quarter, month, season, holidays, and temperature categories.
    """, icon="🚩")
    with st.container():
        In_0, In_1, In_2 = st.columns([150,80,80])
        with In_0:
            st.subheader("Over view Monthly Transactions Count/Year")
            Cross_tab =pd.crosstab(index=df_new['year'] , columns=df_new['month'] , values=df_new['month'] , aggfunc= 'count',margins=True).reset_index().replace(np.nan,0)
            st.dataframe(Cross_tab , column_config={"year": st.column_config.TextColumn("year"),"All": st.column_config.ProgressColumn("All",format="%f",min_value=0,max_value=max(Cross_tab.All),)},hide_index=True,use_container_width=True)
        with In_1:
            st.subheader("Over view weekly_sales/Year")
            Years_Sales = df_new.groupby(['year'])['weekly_sales'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(Years_Sales,column_order=("year", "weekly_sales"),
                         hide_index=True,width=None,use_container_width=True,
                         column_config={"year":st.column_config.TextColumn("year"),"weekly_sales":st.column_config.ProgressColumn("weekly_sales",format="%f", min_value=0, max_value=max(Years_Sales.weekly_sales),)})
        with In_2:
            st.subheader("Over view weekly_sales/season")
            Season_Sales = df_new.groupby(['season'])['weekly_sales'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(Season_Sales,column_order=("season", "weekly_sales"), 
                        hide_index=True,width=None,use_container_width=True,
                        column_config={"season": st.column_config.TextColumn("season"),"weekly_sales": st.column_config.ProgressColumn("weekly_sales",format="%f",min_value=0,max_value=max(Season_Sales.weekly_sales),)})    
    with st.container():
        with In_0:
            st.success("""
                * It becomes evident that:
                * 2011 emerges as the top sales year, boasting the highest figures.
                    * Although 2012 shows lower sales, it's worth noting that data for the entire year isn't available.
                    * The absence of data for January 2010 may impact insights for that year.
                * Overall, summer emerges as the peak season for sales.
                * Store No. 20 ranks highest in sales over the three-year period.
                * Conversely, Store No. 33 exhibits lower sales across the same timeframe.
                * Sales during holidays are lower compared to regular working days, suggesting that customers may utilize holidays for activities other than making purchases.
                * Moderate weather conditions appear to be optimal for sales, with both cold and warm weather showing similar levels of sales.""")
        with In_1:
            st.subheader("Summary")
            Total_Sales = df_new.weekly_sales.sum()
            AVG_Sales = df_new.weekly_sales.mean()
            Total_Transactions = df_new.weekly_sales.count()
            Total_Stores = df_new.store.nunique()
            st.metric(label="Total Sales", value=round(Total_Sales,2))
            st.metric(label="AVG_Sales", value=round(AVG_Sales,2))
            st.metric(label="Total_Transactions", value=round(Total_Transactions,2))
            st.metric(label="Total_Stores", value=round(Total_Stores,2))   
        with In_2:
            st.subheader("Over view weekly_sales/Store")
            Store_Sales = df_new.groupby(['store'])['weekly_sales'].sum().sort_values(ascending=False).reset_index()
            st.dataframe(Store_Sales,column_order=("store", "weekly_sales"),
                         hide_index=True,width=None,use_container_width=True,
                         column_config={"store":st.column_config.TextColumn("store"),"weekly_sales":st.column_config.ProgressColumn("weekly_sales",format="%f", min_value=0, max_value=max(Store_Sales.weekly_sales),)})
    with st.container():
        In_3, In_4, In_5 , In_6 = st.columns([60,70,70,70])
        Main_Ins_Data = df_new.groupby(['year','quarter','month','season','holiday_flag','temperature_class'])[['fuel_price','cpi','unemployment','weekly_sales']].mean().reset_index()
        Fuel_Q = px.histogram(Main_Ins_Data , x='quarter' , y='fuel_price',color='year' , text_auto=True , barmode='group', title='AVG fuel_price/quarter')
        CPI_Q = px.histogram(Main_Ins_Data , x='quarter' , y='cpi',color='year' , text_auto=True , barmode='group', title='AVG cpi/quarter')
        UnEmp_Q = px.histogram(Main_Ins_Data , x='quarter' , y='unemployment',color='year' , text_auto=True , barmode='group', title='AVG unemployment/quarter')
        
        Sales_month = px.histogram(Main_Ins_Data , x='month' , y='weekly_sales',color='year' , text_auto=True , barmode='group', title='AVG weekly_sales/month')
        Sales_season = px.histogram(Main_Ins_Data , x='season' , y='weekly_sales',color='year' , text_auto=True , barmode='group', title='AVG weekly_sales/season')
        Sales_holiday_flag = px.histogram(Main_Ins_Data , x='holiday_flag' , y='weekly_sales',color='year' , text_auto=True , barmode='group', title='AVG weekly_sales/holiday_flag')
        Sales_temperature_class = px.histogram(Main_Ins_Data , x='temperature_class' , y='weekly_sales',color='year' , text_auto=True , barmode='group', title='AVG weekly_sales/temperature_class')
        
        In_5_2 = px.histogram(Main_Ins_Data , x='quarter' , y='weekly_sales',color='year' , text_auto=True , barmode='group', title='AVG weekly_sales/quarter')
        with In_3:
            st.plotly_chart(Fuel_Q , use_container_width=True)
            st.plotly_chart(Sales_month , use_container_width=True)
        with In_4: # fuel_price
            st.plotly_chart(CPI_Q , use_container_width=True)
            st.plotly_chart(Sales_season , use_container_width=True)
        with In_5: # cpi
            st.plotly_chart(UnEmp_Q , use_container_width=True)
            st.plotly_chart(Sales_holiday_flag , use_container_width=True)
        with In_6: # unemployment
            st.plotly_chart(In_5_2 , use_container_width=True)
            st.plotly_chart(Sales_temperature_class , use_container_width=True)
        
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

with Dashboard:
    st.header('InteractiveDashboard')
    with st.container():
        #############################################################################
        # Define Custom DF fillter
        def Fillter_Data (df , store , year ):
            if store == 'all':
                store = df['store'].unique().tolist()
            elif store != 'all':
                store = store
            # Year
            if year == 'all':
                year = df['year'].unique().tolist()
            elif year != 'all':
                year = year
                
            return df[(df['store'].isin(store)) & 
                    (df['year'].isin(year))].sort_values(by = 'date',ascending = True)
        #############################################################################
        Dash_1 , S_1 ,Dash_2, S_2 , Dash_3 = st.columns([50,5,50,3,20])
        Store = None
        Years = None
        Grouping = None
        Coloring = None
        Calculation = None
        with Dash_1:
            st.subheader("Store")
            DB_x = st.radio("Store:",["All" , "Custom"])
            if DB_x == "All":
                Store = df_new.store.unique().tolist()
            else:
                Store = st.multiselect("Select Store:",df_new.store.unique().tolist())
        with Dash_2:
            st.subheader("Years")
            DB_x_1 = st.radio("Years:",["All","Custom"])
            if DB_x_1 == "All":
                Years = df_new.year.unique().tolist()
            else:
                Years = st.multiselect("Select Year:",df_new.year.unique().tolist())
        with Dash_3:
            st.subheader("Coloring")
            Coloring = st.radio("Coloring:",["year","holiday_flag"])

    with st.container():
        Custom_Df = Fillter_Data (df_new , Store , Years )
        Vis_1 , Vis_2 , Vis_3  , Vis_4 = st.columns([50,50,50,50])
        with Vis_1:
            st.success(f"AVG Monthly (fuel_price) per ({Coloring})")
            VisuData_1 = Custom_Df.groupby(['month',Coloring])['fuel_price'].mean().reset_index()
            Visu_1 = px.line(VisuData_1 , x='month' , y='fuel_price' , color = Coloring)
            st.plotly_chart(Visu_1 , use_container_width=True)
        with Vis_2:
            st.success(f"AVG Monthly (cpi) per ({Coloring})")
            VisuData_2 = Custom_Df.groupby(['month',Coloring])['cpi'].mean().reset_index()
            Visu_2 = px.line(VisuData_2 , x='month' , y='cpi' , color = Coloring)
            st.plotly_chart(Visu_2 , use_container_width=True)        
        with Vis_3:
            st.success(f"AVG Monthly (unemployment) per ({Coloring})")
            VisuData_3 = Custom_Df.groupby(['month',Coloring])['unemployment'].mean().reset_index()
            Visu_3 = px.line(VisuData_3 , x='month' , y='unemployment' , color = Coloring)
            st.plotly_chart(Visu_3 , use_container_width=True)
        with Vis_4:
            st.success(f"AVG Monthly (weekly_sales) per ({Coloring})")
            VisuData_4 = Custom_Df.groupby(['month',Coloring])['weekly_sales'].mean().reset_index()
            Visu_4 = px.line(VisuData_4 , x='month' , y='weekly_sales' , color = Coloring)
            st.plotly_chart(Visu_4 , use_container_width=True)    
    with st.container():
        with Vis_1:
            st.success(f"Transactions Count per month/({Coloring})")
            VisuData_5 = Custom_Df.groupby(['month',Coloring])['store'].count().reset_index()
            Visu_5 = px.bar(VisuData_5 , x='month' , y='store' , color = Coloring , barmode='group',text_auto = True)
            st.plotly_chart(Visu_5 , use_container_width=True) 
        with Vis_2:
            st.success(f"Transactions Count per quarter/({Coloring})")
            VisuData_6 = Custom_Df.groupby(['quarter',Coloring])['store'].count().reset_index()
            Visu_6 = px.histogram(VisuData_6 , x='quarter' , y='store' , color = Coloring , barmode='group',text_auto = True)
            st.plotly_chart(Visu_6 , use_container_width=True)
        with Vis_3:
            st.success(f"Transactions Count per season/({Coloring})")
            VisuData_7 = Custom_Df.groupby(['season',Coloring])['store'].count().reset_index()
            Visu_7 = px.histogram(VisuData_7 , x='season' , y='store' , color = Coloring , barmode='group',text_auto = True)
            st.plotly_chart(Visu_7 , use_container_width=True)        
        with Vis_4:
            st.success(f"Transactions Count per temperature_class/({Coloring})")
            VisuData_8 = Custom_Df.groupby(['temperature_class',Coloring])['store'].count().reset_index()
            Visu_8 = px.histogram(VisuData_8 , x='temperature_class' , y='store' , color = Coloring , barmode='group',text_auto = True)
            st.plotly_chart(Visu_8 , use_container_width=True)
    with st.container():
        with Vis_1:
            st.success(f"Total weekly_sales per month/({Coloring})")
            VisuData_9 = Custom_Df.groupby(['month',Coloring])['weekly_sales'].sum().reset_index()
            Visu_9 = px.bar(VisuData_9 , x='month' , y='weekly_sales' , color = Coloring , barmode='group',text_auto = True,color_discrete_sequence= px.colors.qualitative.Dark24)
            st.plotly_chart(Visu_9 , use_container_width=True) 
        with Vis_2:
            st.success(f"Total weekly_sales per quarter/({Coloring})")
            VisuData_10 = Custom_Df.groupby(['quarter',Coloring])['weekly_sales'].sum().reset_index()
            Visu_10 = px.histogram(VisuData_10 , x='quarter' , y='weekly_sales' , color = Coloring , barmode='group',text_auto = True,color_discrete_sequence= px.colors.qualitative.Dark24)
            st.plotly_chart(Visu_10 , use_container_width=True)
        with Vis_3:
            st.success(f"Total weekly_sales per season/({Coloring})")
            VisuData_11 = Custom_Df.groupby(['season',Coloring])['weekly_sales'].sum().reset_index()
            Visu_11 = px.histogram(VisuData_11 , x='season' , y='weekly_sales' , color = Coloring , barmode='group',text_auto = True,color_discrete_sequence= px.colors.qualitative.Dark24)
            st.plotly_chart(Visu_11 , use_container_width=True)        
        with Vis_4:
            st.success(f"Total weekly_sales per temperature_class/({Coloring})")
            VisuData_12 = Custom_Df.groupby(['temperature_class',Coloring])['weekly_sales'].sum().reset_index()
            Visu_12 = px.histogram(VisuData_12 , x='temperature_class' , y='weekly_sales' , color = Coloring , barmode='group',text_auto = True,color_discrete_sequence= px.colors.qualitative.Dark24)
            st.plotly_chart(Visu_12 , use_container_width=True)
# ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
with Prediction:
    st.header('Prediction')
    with st.container():
        PRD_00 , PRD_01 , PRD_02 = st.columns([50,5,50])
        with PRD_00:
            Selected_Date = st.date_input("Date:" , min_value = pd.Timestamp( "2000-01-01" ) , max_value = pd.Timestamp( "2100-12-31" ) , value=pd.Timestamp.today())
            store = st.selectbox("store:" , df_new.store.unique().tolist())
            temperature = st.slider("temperature:" , min(df_new['temperature']) , max(df_new['temperature']))
            fuel_price = st.slider("fuel_price:" , min(df_new['fuel_price']) , max(df_new['fuel_price']))

            year = Selected_Date.year
            month = Selected_Date.month
            day = Selected_Date.strftime("%A") 
            
        with PRD_02:
            # holiday_flag
            holiday_flag = st.radio ("holiday_flag:" , ('holiday' , 'Not holiday'))
            cpi = st.slider("cpi:" , min(df_new['cpi']) , max(df_new['cpi']))
            unemployment = st.slider("unemployment:" , min(df_new['unemployment']) , max(df_new['unemployment']))
            if holiday_flag == 'holiday':
                holiday_flag =  '1'
            else:
                holiday_flag =  '0'
                
            # month
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            elif month in [9, 10, 11]:
                season = 'Fall'
            
            N_date = pd.DataFrame({'store':store,'year':year,
                                   'month':month,'day':day,'season':season,
                                   'holiday_flag':holiday_flag,'temperature':temperature,
                                   'fuel_price':fuel_price,'cpi':cpi,'unemployment':unemployment} , index=[0])
            st.write(N_date)

            Transformer = jb.load('preprocessor.h5')
            Model = jb.load('XGBR.h5')
            TEST_Procc = Transformer.transform(N_date)
            TEST_PRED = Model.predict(TEST_Procc)
            if st.button('Predict'):
                st.header(f"weekly_sales : {float(round(TEST_PRED[0],2))}")
                st.balloons()
        


        
