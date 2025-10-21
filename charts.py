import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
df1=pd.read_csv("covid.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
#print(df1.head())
#print(df1.tail())
#print(df1.shape)
#print(df1.sample(9))
#print(df1.info())
#print(df1.columns)
#print(df1.nunique().sort_values(ascending=False).head(17))
# Drop NewCases, NewDeaths, NewRecovered rows from dataset1
df1.drop(['NewCases', 'NewDeaths', 'NewRecovered'],
              axis=1, inplace=True)

# Step 2: Select a few columns (optional, just to make the table readable)
data =df1

# Step 3: Create a Plotly Table
col_widths = [max(data[col].astype(str).map(len).max(), len(col)) * 12 for col in data.columns]
#print(col_widths)
row_colors = ['aliceblue', 'whitesmoke'] * (len(data) // 2 + 1)
fig = go.Figure(
    data=[go.Table(
        columnwidth=col_widths,
        header=dict(
            values=[f"<b>{col}</b>" for col in data.columns],  # Bold headers
            fill_color='navy',  # Header background color
            align='center',
            font=dict(color='white', size=13),
            line_color='lightgray',
            height=35
        ),
        cells=dict(
            values=[data[col] for col in data.columns],
            fill_color=[row_colors],  # Alternate row colors
            align='center',
            font=dict(color='dimgray', size=12),
            line_color='lightgray',  # Black border between cells
            height=28
        )
    )]
)

# Step 4: Add a title
fig.update_layout(
    width=sum(col_widths) + 200,
    title_text='COVID-19 Dataset',
    title_x=0.5,  # Center title
    title_font=dict(size=20, color='midnightblue')
)

# Step 5: Display the table
#fig.show()
#fig.write_html("COVID 19 DATASET SUMMARY TABLE .html")


fig = px.bar(df1.nlargest(15, 'TotalCases'),
             x='Country/Region',
             y='TotalCases',
             color='TotalCases',
             hover_data=['Country/Region', 'Continent'],
             title='Top 15 Countries by Total COVID-19 Cases')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("Top 15 Countries by Total COVID-19 Cases.html")

fig_totalcasesdeaths = px.bar(df1.nlargest(15, 'TotalCases'),
             x='Country/Region',
             y='TotalCases',
             color='TotalDeaths',
             hover_data=['Country/Region', 'Continent','TotalRecovered','TotalTests','WHO Region'],
             title='Top 15 Countries by Total COVID-19 Cases and deaths')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("Top 15 Countries by Total COVID-19 Cases and deaths.html")

fig = px.bar(df1.nlargest(15, 'TotalCases'),
             x='Country/Region',
             y='TotalCases',
             color='TotalRecovered',
             hover_data=['Country/Region', 'Continent'],
             title='Top 15 Countries by Total COVID-19 Cases and recoveries')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("Top 15 Countries by Total COVID-19 Cases and recoveries.html")

fig = px.bar(df1.nlargest(15, 'TotalCases'),
             x='Country/Region',
             y='TotalCases',
             color='TotalTests',
             hover_data=['Country/Region', 'Continent'],
             title='Top 15 Countries by Total COVID-19 Cases and total tests')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("Top 15 Countries by Total COVID-19 Cases and total tests.html")

fig = px.bar(df1.nlargest(15, 'TotalCases'),
             x='TotalTests',
             y='Country/Region',
             color='TotalTests',
             hover_data=['Country/Region', 'Continent','TotalRecovered','WHO Region'],
             orientation="h",
             title='Top 15 Countries by total tests ')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("Top 15 Countries total tests.html")

fig = px.bar(df1.nlargest(15, 'TotalCases'),
             x='TotalTests',
             y='Continent',
             color='TotalTests',
             hover_data=['Country/Region','WHO Region'],
             orientation="h",
             title='Top 15 Countries by total tests ')
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("continent by  total tests.html")

fig_scatterplot=px.scatter(df1, x='Continent',y='TotalCases',
           hover_data={'Country/Region':True,'WHO Region':True,'Continent':True,'TotalCases': True},
           title='scatter plot of continent by  total cases',
           color='TotalRecovered',size='TotalCases',size_max=80)
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("scatter plot of continent by  total cases.html")

fig=px.scatter(df1, x='Continent',y='TotalCases',
           hover_data=['Country/Region', 'Continent'],
           title='scatter plot of continent by  total cases log',
           color='TotalCases', size='TotalCases', size_max=80, log_y=True)
fig.update_layout(title_x=0.5)
#fig.show()
#fig.write_html("scatter plot of continent by  total cases log.html")

#makng an interactive table for continent data
continent_summary = df1.groupby('Continent', as_index=False)[
    ['TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases','TotalTests']
].sum()
continent_summary['DeathRate (%)'] = (continent_summary['TotalDeaths'] / continent_summary['TotalCases']) * 100
continent_summary['DeathRate (%)']=continent_summary['DeathRate (%)'].round(2)
total_row = pd.DataFrame({
    'Continent': ['World Total'],
    'TotalCases': [df1['TotalCases'].sum()],
    'TotalDeaths': [df1['TotalDeaths'].sum()],
    'TotalRecovered': [df1['TotalRecovered'].sum()],
    'ActiveCases': [df1['ActiveCases'].sum()],
    'TotalTests': [df1['TotalTests'].sum()],
    'DeathRate (%)': [round((df1['TotalDeaths'].sum() / df1['TotalCases'].sum()) * 100,2)]
})
continent_summary = continent_summary.sort_values(by='TotalCases', ascending=False)
continent_summary = pd.concat([continent_summary, total_row], ignore_index=True)
cols_to_format = ['TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases','TotalTests']
for col in cols_to_format:
    continent_summary[col] = continent_summary[col].apply(lambda x: f"{x:,}")
row_colors=['lightyellow','lavender']
fig_contable = go.Figure(
    data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in continent_summary.columns],  # Bold headers
            fill_color='royalBlue',  # Header background color
            align='center',
            font=dict(color='white', size=13),
            line_color='purple',
            height=35
        ),
        cells=dict(
            values=[continent_summary[col] for col in continent_summary.columns],
            fill_color=[row_colors * (len(continent_summary) // 2 + 1)],  # Alternate row colors
            align='center',
            font=dict(color='darkBlue', size=12),
            line_color='purple',  # Black border between cells
            height=28
        )
    )]
)

# Step 4: Add a title
fig_contable.update_layout(
    title_text='COVID-19 Dataset by Continent',
    title_x=0.5,  # Center title
    title_font=dict(size=20, color='midnightblue'),
    width=1100,   # Increase overall figure width
    height=600,   # Increase figure height so last row shows well
    margin=dict(l=40, r=40, t=80, b=40)  # Add some spacing around
)


# Step 5: Display the table
#fig_contable.show()
#fig_contable.write_html("COVID 19 DATASET BY continent summary .html")

df1['TotalDeaths'] = df1['TotalDeaths'].fillna(0)
fig_impact=px.scatter(df1, x='Continent',y='Population',
           hover_data=['Country/Region', 'WHO Region', 'TotalCases'],
           title='Impact of covid 19 deaths relative to population',
           color='TotalDeaths', size='TotalDeaths', size_max=80, log_y=False)
fig_impact.update_layout(title_x=0.5)
#fig_impact.show()
#fig_impact.write_html("Impact of covid 19 deaths relative to population.html")

# Group by WHO region and sum up total cases and recoveries
df_region = df1.groupby('WHO Region', as_index=False)[['TotalCases', 'TotalRecovered']].sum()

# Calculate recovery rate (%)
df_region['RecoveryRate (%)'] = (df_region['TotalRecovered'] / df_region['TotalCases']) * 100

df_region = df_region.sort_values(by='TotalCases', ascending=False)

# Melt the dataframe for side-by-side bars
df_melted = df_region.melt(
    id_vars=['WHO Region', 'RecoveryRate (%)'],
    value_vars=['TotalCases', 'TotalRecovered'],
    var_name='Category',
    value_name='Count'
)

# Create side-by-side bar chart
fig_tcasesbyrecov = px.bar(
    df_melted,
    x='WHO Region',
    y='Count',
    color='Category',
    text='Count',
    barmode='group',  # side-by-side bars
    title='Total Cases and Recoveries by WHO Region',
    hover_data={'WHO Region': True, 'RecoveryRate (%)': ':.2f', 'Count': ':,'},  # hover details
    color_discrete_sequence=['darkblue', 'gold']  # blue for cases, green for recovered
)

# Add formatting
fig_tcasesbyrecov.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
fig_tcasesbyrecov.update_layout(
    xaxis_title='WHO Region',
    yaxis_title='Number of Cases',
    title_x=0.5,
    legend_title_text='Category',
    bargap=0.2
)

#fig_tcasesbyrecov.show()


# Group by WHO region and sum total cases and recoveries
df_region = df1.groupby('WHO Region', as_index=False)[['TotalCases', 'TotalRecovered']].sum()

# Calculate recovery rate (%)
df_region['RecoveryRate (%)'] = (df_region['TotalRecovered'] / df_region['TotalCases']) * 100

# Sort by recovery rate (highest to lowest)
df_region = df_region.sort_values(by='RecoveryRate (%)', ascending=False)

# Create horizontal bar chart
fig_recoverate = px.bar(
    df_region,
    y='WHO Region',
    x='RecoveryRate (%)',
    orientation='h',
    text='RecoveryRate (%)',
    title='Recovery Rate by WHO Region',
    color='RecoveryRate (%)',
    color_continuous_scale=['darkblue', 'gold'],  # deep blue to bright yellow gradient
    hover_data={
        'WHO Region': True,
        'TotalCases': ':,',
        'TotalRecovered': ':,',
        'RecoveryRate (%)': ':.2f'
    }
)

# Format text and layout
fig_recoverate.update_traces(
    texttemplate='%{text:.2f}%',
    textposition='outside'
)
fig_recoverate.update_layout(
    xaxis_title='Recovery Rate (%)',
    yaxis_title='WHO Region',
    title_x=0.5,
    coloraxis_colorbar=dict(title='Recovery Rate (%)'),
    height=450,
    bargap=0.3
)

#fig_recoverate.show()
