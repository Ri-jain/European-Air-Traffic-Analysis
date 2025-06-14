#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aviation Traffic Analysis (2016-2024)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

# ==============================================
# Aviation Data Visualization Theme (Compact)
# ==============================================
# Define color palettes for consistent visualization
period_colors = {
    'Pre-Pandemic': '#1A5276',  # Deep blue
    'Pandemic': '#C0392B',      # Deep red
    'Post-Pandemic': '#27AE60'  # Emerald green
}
# Year-specific palette with intuitive progression
year_palette = {
    2016: '#1F4788', 2017: '#2E6BAC', 2018: '#3E8ECC', 2019: '#5DB1EC',  # Blues
    2020: '#C0392B', 2021: '#E74C3C',  # Reds
    2022: '#229954', 2023: '#27AE60', 2024: '#2ECC71'  # Greens
}
# Country-specific colors for major European nations
country_colors = {
    'United Kingdom': '#CF142B', 'Germany': '#000000', 'France': '#0055A4',
    'Spain': '#FFC400', 'Italy': '#008C45'
}
# Airport category colors
airport_category_colors = {
    'leisure': '#FF7700', 'business': '#4A235A', 'other': '#808B96'
}
# Configure the overall theme settings
def set_aviation_theme():
    """Apply the aviation theme to all subsequent plots"""
    # Figure aesthetics
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['figure.dpi'] = 100
    
    # Text properties
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 11
    
    # Grid properties
    mpl.rcParams['grid.color'] = '#E5E8E8'
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = 0.5
    
    # Legend properties
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.9
    mpl.rcParams['legend.fancybox'] = True
    
    # Tick appearance
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['ytick.major.size'] = 4
    
    # Savefig settings for exports
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    
    # Seaborn settings
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_context("talk", font_scale=0.8)

# Function to add a COVID reference to plots
def add_covid_reference(ax, color='#C0392B', alpha=0.2):
    """Add a COVID period highlight to a time series plot"""
    # Add vertical line at 2020
    ax.axvline(x=2020, color=color, linestyle='--', alpha=0.7, label='COVID-19 Pandemic')
    
    # Add shaded area for 2020-2021
    ax.axvspan(2020, 2021, color=color, alpha=alpha, label='Pandemic Period')

# Apply the theme
set_aviation_theme()

# ==============================================
# Data Loading and Preparation
# ==============================================

# Load data
try:
    merge_pd = pd.read_csv('airport_traffic_2016_2024_cleaned.csv')
    print(f"Data loaded successfully with shape: {merge_pd.shape}")
except FileNotFoundError:
    print("Error: File 'airport_traffic_2016_2024_cleaned.csv' not found.")
    raise

# Check for null values
null_counts = merge_pd.isnull().sum()
print("\nNull value counts:")
print(null_counts[null_counts > 0] if any(null_counts > 0) else "No null values found")

# Check the years in the dataset
years_in_dataset = merge_pd['YEAR'].unique()
print(f"\nYears in dataset: {sorted(years_in_dataset)}")

# Define pandemic periods for reference
pre_pandemic = [2016, 2017, 2018, 2019]
pandemic = [2020, 2021]
post_pandemic = [2022, 2023, 2024]

# ==============================================
# Question 1: Overall Traffic Trends
# ==============================================

# Part 1: Total Flight Volume Across Europe
yearly_traffic = merge_pd.groupby('YEAR')[['FLT_DEP_1', 'FLT_ARR_1', 'FLT_TOT_1']].sum().reset_index()

plt.figure()
plt.plot(yearly_traffic['YEAR'], yearly_traffic['FLT_TOT_1'], marker='o', 
         color='#3498DB', linewidth=2, markersize=8)
add_covid_reference(plt.gca())
plt.title('Total Flight Volume Across Europe (2016–2024)')
plt.xlabel('Year')
plt.ylabel('Total Flights')
plt.grid(True)
plt.tight_layout()
plt.show()

# Part 2: Seasonal Patterns Analysis
monthly_trends = merge_pd.groupby(['YEAR', 'MONTH_NUM'])[['FLT_DEP_1', 'FLT_ARR_1']].sum().reset_index()

# Departures with year-based color palette
plt.figure()
for year in sorted(monthly_trends['YEAR'].unique()):
    year_data = monthly_trends[monthly_trends['YEAR'] == year]
    plt.plot(year_data['MONTH_NUM'], year_data['FLT_DEP_1'], 
             marker='o' if year in [2019, 2020, 2024] else None,
             label=str(int(year)), color=year_palette.get(int(year), '#333333'),
             linewidth=2 if year in [2019, 2020, 2024] else 1.5,
             alpha=0.8 if year in [2019, 2020, 2024] else 0.6)

plt.title('Seasonal Patterns in Flight Departures (2016–2024)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Departures')
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Arrivals with year-based color palette
plt.figure()
for year in sorted(monthly_trends['YEAR'].unique()):
    year_data = monthly_trends[monthly_trends['YEAR'] == year]
    plt.plot(year_data['MONTH_NUM'], year_data['FLT_ARR_1'], 
             marker='o' if year in [2019, 2020, 2024] else None,
             label=str(int(year)), color=year_palette.get(int(year), '#333333'),
             linewidth=2 if year in [2019, 2020, 2024] else 1.5,
             alpha=0.8 if year in [2019, 2020, 2024] else 0.6)

plt.title('Seasonal Patterns in Flight Arrivals (2016–2024)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Arrivals')
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ==============================================
# Question 2: Country-Level Analysis
# ==============================================

# Part 1: Year-over-Year Growth for Top Countries
traffic_by_year = merge_pd.groupby(['STATE_NAME', 'YEAR'])['FLT_TOT_1'].sum().reset_index()
traffic_by_year.rename(columns={'FLT_TOT_1': 'Total_Flights'}, inplace=True)

# Calculate year-over-year growth
traffic_by_year['YoY_Growth'] = traffic_by_year.groupby('STATE_NAME')['Total_Flights'].pct_change() * 100

traffic_pivot = traffic_by_year.pivot(index='STATE_NAME', columns='YEAR', values='Total_Flights')
growth_pivot = traffic_by_year.pivot(index='STATE_NAME', columns='YEAR', values='YoY_Growth')

# Get top 10 countries by total flights
top10_countries_overall = traffic_pivot.sum(axis=1).sort_values(ascending=False).head(10).index.tolist()

# Plot YoY Growth Trends for top 10 countries with custom colors
plt.figure()
for i, country in enumerate(top10_countries_overall):
    if country in growth_pivot.index:
        country_data = growth_pivot.loc[country].dropna()
        if not country_data.empty:
            years = country_data.index.astype(int)
            values = country_data.values
            color = country_colors.get(country, plt.cm.tab10(i % 10))
            plt.plot(years, values, marker='o', linewidth=2, markersize=6, 
                     label=country, color=color)

add_covid_reference(plt.gca())
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Zero reference line
plt.title("YoY Growth Trends for Top 10 Countries (2016–2024)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("YoY Growth (%)")
plt.grid(True)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Part 2: UK vs Major EU Countries Comparison
major_countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy']

uk_vs_eu = merge_pd[merge_pd['STATE_NAME'].isin(major_countries)]
uk_trends = uk_vs_eu.groupby(['STATE_NAME', 'YEAR'])['FLT_TOT_1'].sum().reset_index()

plt.figure()
for country in major_countries:
    data = uk_trends[uk_trends['STATE_NAME'] == country]
    color = country_colors.get(country, 'gray')
    plt.plot(data['YEAR'], data['FLT_TOT_1'], label=country, marker='o', 
             color=color, linewidth=2, markersize=6)

add_covid_reference(plt.gca())
plt.title('Flight Operations: UK vs Major EU Countries (2016–2024)')
plt.xlabel('Year')
plt.ylabel('Total Flights')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Part 3: Post-COVID Recovery Analysis by Country
years_available = merge_pd['YEAR'].unique()
if 2019 in years_available and 2023 in years_available:
    pre_covid = merge_pd[merge_pd['YEAR'] == 2019].groupby('STATE_NAME')['FLT_TOT_1'].sum().reset_index(name='Flights_2019')
    post_covid = merge_pd[merge_pd['YEAR'] == 2023].groupby('STATE_NAME')['FLT_TOT_1'].sum().reset_index(name='Flights_2023')

    recovery_df = pd.merge(pre_covid, post_covid, on='STATE_NAME')
    recovery_df['Recovery_%'] = (recovery_df['Flights_2023'] / recovery_df['Flights_2019']) * 100
    recovery_df = recovery_df.sort_values(by='Recovery_%', ascending=False)

    # Show only top 15 recovered countries
    top_recovery = recovery_df.head(15)

    plt.figure()
    # Create a sequential color palette based on recovery percentage
    recovery_colors = []
    for pct in top_recovery['Recovery_%']:
        if pct >= 110:  # Strong growth
            recovery_colors.append('#1E8449')  # Dark green
        elif pct >= 100:  # Full recovery
            recovery_colors.append('#27AE60')  # Medium green
        elif pct >= 90:  # Near recovery
            recovery_colors.append('#F1C40F')  # Yellow
        elif pct >= 80:  # Significant progress
            recovery_colors.append('#E67E22')  # Orange
        else:  # Lagging
            recovery_colors.append('#C0392B')  # Red
            
    sns.barplot(data=top_recovery, x='Recovery_%', y='STATE_NAME', 
                palette=recovery_colors, hue=None)
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='2019 Level')
    plt.title('Top 15 Post-COVID Air Traffic Recoveries (2023 vs 2019)')
    plt.xlabel('% Recovery')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()

# ==============================================
# Question 3: Airport Performance & Ranking
# ==============================================

# 3.1 Which airports grew the most from 2016 to 2024?
if 2016 in years_available and 2024 in years_available:
    # Group by airport and year, and sum flight metrics
    airport_yearly = merge_pd.groupby(['APT_ICAO', 'APT_NAME', 'STATE_NAME', 'YEAR'])['FLT_TOT_1'].sum().reset_index()

    # Create a pivot table for easier comparison
    airport_pivot = airport_yearly.pivot_table(
        index=['APT_ICAO', 'APT_NAME', 'STATE_NAME'], 
        columns='YEAR', 
        values='FLT_TOT_1',
        fill_value=0
    ).reset_index()

    # Make sure the required years are in the columns
    if 2016 in airport_pivot.columns and 2024 in airport_pivot.columns:
        # Calculate growth between 2016 and 2024
        # Only include airports with data for both years
        airport_growth = airport_pivot[airport_pivot[2016] > 0].copy()
        airport_growth['growth_absolute'] = airport_growth[2024] - airport_growth[2016]
        airport_growth['growth_percentage'] = (airport_growth[2024] / airport_growth[2016] - 1) * 100

        # Top 15 airports by absolute growth
        top_growth = airport_growth.sort_values('growth_absolute', ascending=False).head(15)
        bottom_growth = airport_growth.sort_values('growth_absolute').head(15)

        # Visualization: Top airports by growth
        plt.figure()
        # Create a sequential color palette based on growth magnitude
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_growth)))
        
        sns.barplot(x='growth_absolute', y='APT_NAME', data=top_growth, 
                   palette=colors, hue='STATE_NAME', dodge=False)
        plt.title('Top 15 Airports by Growth in Total Flights (2016-2024)')
        plt.xlabel('Growth in Total Flights')
        plt.ylabel('Airport')
        plt.tight_layout()
        plt.show()
        
        # Visualization: Bottom airports by growth (most declined)
        plt.figure()
   
        
        sns.barplot(x='growth_absolute', y='APT_NAME', data=bottom_growth, 
                   palette=colors, hue='STATE_NAME', dodge=False)
        plt.title('15 Airports with Largest Decline in Total Flights (2016-2024)')
        plt.xlabel('Decline in Total Flights')
        plt.ylabel('Airport')
        plt.tight_layout()
        plt.show()

# 3.2 Percentage of total traffic handled by top 10 busiest airports over time
total_traffic_by_year = merge_pd.groupby('YEAR')['FLT_TOT_1'].sum().reset_index()

# Calculate traffic for top 10 airports by year
top_airports_by_year = {}
top10_traffic_by_year = []

for year in merge_pd['YEAR'].unique():
    year_data = merge_pd[merge_pd['YEAR'] == year]
    if not year_data.empty:
        top_airports = year_data.groupby(['APT_ICAO'])['FLT_TOT_1'].sum().nlargest(10).index.tolist()
        top_airports_by_year[year] = top_airports
        
        traffic = year_data[year_data['APT_ICAO'].isin(top_airports)]['FLT_TOT_1'].sum()
        top10_traffic_by_year.append({'YEAR': year, 'TOP10_TRAFFIC': traffic})

top10_df = pd.DataFrame(top10_traffic_by_year)

# Merge with total traffic and calculate percentage
traffic_concentration = pd.merge(top10_df, total_traffic_by_year, on='YEAR')
traffic_concentration['TOP10_PERCENTAGE'] = (traffic_concentration['TOP10_TRAFFIC'] / traffic_concentration['FLT_TOT_1']) * 100

plt.figure()
# Create scatter points with year-based colors
for i, year in enumerate(traffic_concentration['YEAR']):
    plt.scatter(year, traffic_concentration.iloc[i]['TOP10_PERCENTAGE'], 
               s=80, color=year_palette.get(int(year), '#333333'), 
               zorder=3, edgecolor='white', linewidth=1)

# Connect with spline for smoother look
plt.plot(traffic_concentration['YEAR'], traffic_concentration['TOP10_PERCENTAGE'], 
         '-', color='#34495E', linewidth=2, alpha=0.7, zorder=2)

add_covid_reference(plt.gca())
plt.title('Percentage of Total Traffic Handled by Top 10 Airports')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Traffic (%)')
plt.grid(True, alpha=0.3)
plt.xticks(traffic_concentration['YEAR'])
plt.tight_layout()
plt.show()

# ==============================================
# Question 4: Commercial vs. Business Flight Trends
# ==============================================

# 4.1 Compare IFR flights to total flights over time
if 'FLT_TOT_IFR_2' in merge_pd.columns:
    flight_types_yearly = merge_pd.groupby('YEAR').agg({
        'FLT_TOT_1': 'sum',
        'FLT_TOT_IFR_2': 'sum'
    }).reset_index()

    flight_types_yearly['IFR_PERCENTAGE'] = (flight_types_yearly['FLT_TOT_IFR_2'] / flight_types_yearly['FLT_TOT_1']) * 100
    flight_types_yearly['NON_IFR_PERCENTAGE'] = 100 - flight_types_yearly['IFR_PERCENTAGE']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(flight_types_yearly['YEAR'], flight_types_yearly['FLT_TOT_1'], 'o-', 
             color='#3498DB', linewidth=2, label='Total Flights')
    plt.plot(flight_types_yearly['YEAR'], flight_types_yearly['FLT_TOT_IFR_2'], 'o-', 
             color='#9B59B6', linewidth=2, label='IFR Flights')
    add_covid_reference(plt.gca())
    plt.title('IFR vs Total Flights (2016-2024)')
    plt.xlabel('Year')
    plt.ylabel('Number of Flights')
    plt.grid(True, alpha=0.3)
    plt.xticks(flight_types_yearly['YEAR'])
    plt.legend()

    plt.subplot(1, 2, 2)
    # Create year-colored bars
    bars = plt.bar(flight_types_yearly['YEAR'], flight_types_yearly['IFR_PERCENTAGE'],
                 color=[year_palette.get(int(year), '#333333') for year in flight_types_yearly['YEAR']])
    add_covid_reference(plt.gca())
    plt.title('IFR Flights as Percentage of Total Flights')
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(flight_types_yearly['YEAR'])
    plt.tight_layout()
    plt.show()

    # 4.2 Airport size analysis for business vs commercial
    # Calculate average yearly traffic for each airport
    airport_size = merge_pd.groupby(['APT_ICAO', 'APT_NAME', 'STATE_NAME'])['FLT_TOT_1'].sum().reset_index()
    airport_size['avg_yearly_traffic'] = airport_size['FLT_TOT_1'] / len(merge_pd['YEAR'].unique())

    # Categorize airports
    airport_size['size_category'] = pd.qcut(
        airport_size['avg_yearly_traffic'], 
        q=[0, 0.5, 0.8, 1.0], 
        labels=['Small', 'Medium', 'Large']
    )

    # Merge airport size back to original data
    airport_map = airport_size[['APT_ICAO', 'size_category']].set_index('APT_ICAO').to_dict()['size_category']
    merge_pd['airport_size'] = merge_pd['APT_ICAO'].map(airport_map)

    # Analyze IFR percentage by airport size over time
    size_ifr_trend = merge_pd.groupby(['YEAR', 'airport_size']).agg({
        'FLT_TOT_1': 'sum',
        'FLT_TOT_IFR_2': 'sum'
    }).reset_index()

    size_ifr_trend['IFR_PERCENTAGE'] = (size_ifr_trend['FLT_TOT_IFR_2'] / size_ifr_trend['FLT_TOT_1']) * 100

    # Pivot for visualization
    size_ifr_pivot = size_ifr_trend.pivot(
        index='YEAR', 
        columns='airport_size', 
        values='IFR_PERCENTAGE'
    )

    # Custom colors for airport sizes
    size_colors = {
        'Small': '#FF7700',   # Orange - more likely business aviation 
        'Medium': '#5DADE2',  # Blue - mixed use
        'Large': '#6C3483'    # Purple - predominantly commercial
    }

    plt.figure()
    for size in size_ifr_pivot.columns:
        plt.plot(size_ifr_pivot.index, size_ifr_pivot[size], 'o-', 
                linewidth=2, markersize=6, 
                color=size_colors.get(size, 'gray'), 
                label=f"{size} Airports")
    
    add_covid_reference(plt.gca())
    plt.title('IFR Flight Percentage by Airport Size (2016-2024)')
    plt.xlabel('Year')
    plt.ylabel('IFR Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Airport Size')
    plt.tight_layout()
    plt.show()

    # 4.3 Seasonality patterns by airport size
    monthly_by_size = merge_pd.groupby(['MONTH_NUM', 'airport_size'])['FLT_TOT_1'].sum().reset_index()
    monthly_by_size_pivot = monthly_by_size.pivot(index='MONTH_NUM', columns='airport_size', values='FLT_TOT_1')

    # Normalize to show percentage of annual traffic
    for col in monthly_by_size_pivot.columns:
        monthly_by_size_pivot[col] = monthly_by_size_pivot[col] / monthly_by_size_pivot[col].sum() * 100

    plt.figure()
    for size in monthly_by_size_pivot.columns:
        plt.plot(monthly_by_size_pivot.index, monthly_by_size_pivot[size], 'o-', 
                linewidth=2, markersize=6, 
                color=size_colors.get(size, 'gray'), 
                label=f"{size} Airports")
    
    plt.title('Monthly Traffic Distribution by Airport Size')
    plt.xlabel('Month')
    plt.ylabel('Percentage of Annual Traffic (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Airport Size')
    plt.tight_layout()
    plt.show()

# ==============================================
# Question 5: Post-Pandemic Recovery Analysis
# ==============================================

# 5.1 Compare flight volumes before, during, and after pandemic
yearly_traffic = merge_pd.groupby('YEAR')['FLT_TOT_1'].sum().reset_index()

# Calculate percentage vs. 2019 (last pre-pandemic year)
if 2019 in yearly_traffic['YEAR'].values:
    pre_pandemic_max = yearly_traffic[yearly_traffic['YEAR'] == 2019]['FLT_TOT_1'].values[0]
    yearly_traffic['Pct_vs_2019'] = (yearly_traffic['FLT_TOT_1'] / pre_pandemic_max * 100) - 100

    # Create period labels for visualization
    yearly_traffic['Period'] = yearly_traffic['YEAR'].apply(
        lambda x: 'Pre-Pandemic' if x in pre_pandemic else 
                'Pandemic' if x in pandemic else 'Post-Pandemic'
    )

    plt.figure()
    # Create color-coded bars by period
    bars = plt.bar(yearly_traffic['YEAR'], yearly_traffic['FLT_TOT_1'], 
                color=[period_colors.get(period, '#333333') for period in yearly_traffic['Period']],
                width=0.7, edgecolor='white', linewidth=0.5)
    
    # Add 2019 reference line
    plt.axhline(y=pre_pandemic_max, color='black', linestyle='--', alpha=0.5, label='2019 Level')
    
    # Add labels for percentage change vs 2019
    for i, row in enumerate(yearly_traffic.itertuples()):
        if hasattr(row, 'Pct_vs_2019') and row.YEAR != 2019:
            plt.text(i, row.FLT_TOT_1 + pre_pandemic_max*0.05, 
                    f"{row.Pct_vs_2019:.1f}%", 
                    ha='center', va='bottom', fontsize=9,
                    color='green' if row.Pct_vs_2019 > 0 else 'red')
    
    plt.title('Flight Volumes Before, During, and After Pandemic')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Flights')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(yearly_traffic['YEAR'])
    plt.tight_layout()
    plt.show()

    # 5.2 Which airports recovered fastest?
    if 2019 in years_available and 2024 in years_available:
        airport_2019 = merge_pd[merge_pd['YEAR'] == 2019].groupby(['APT_ICAO', 'APT_NAME', 'STATE_NAME'])['FLT_TOT_1'].sum().reset_index()
        airport_2024 = merge_pd[merge_pd['YEAR'] == 2024].groupby(['APT_ICAO', 'APT_NAME', 'STATE_NAME'])['FLT_TOT_1'].sum().reset_index()

        # Merge to calculate recovery percentages
        recovery_2024 = pd.merge(airport_2019, airport_2024, on=['APT_ICAO', 'APT_NAME', 'STATE_NAME'], suffixes=('_2019', '_2024'))
        recovery_2024['recovery_pct'] = (recovery_2024['FLT_TOT_1_2024'] / recovery_2024['FLT_TOT_1_2019']) * 100

        # Filter for airports with significant traffic
        min_traffic = 10000  # Minimum annual flights to be considered
        recovery_2024_filtered = recovery_2024[recovery_2024['FLT_TOT_1_2019'] > min_traffic]
        
        # Top 15 fastest recovered airports
        top_recovery_2024 = recovery_2024_filtered.sort_values('recovery_pct', ascending=False).head(15)

        plt.figure()
        # Define recovery color function
        def recovery_color(pct):
            if pct >= 120:
                return '#1E8449'  # Dark green
            elif pct >= 100:
                return '#27AE60'  # Medium green
            elif pct >= 90:
                return '#F1C40F'  # Yellow
            elif pct >= 80:
                return '#E67E22'  # Orange
            else:
                return '#C0392B'  # Red
        
        bar_colors = [recovery_color(pct) for pct in top_recovery_2024['recovery_pct']]
        
        # Create horizontal bar chart
        plt.barh(y=top_recovery_2024['APT_NAME'], width=top_recovery_2024['recovery_pct'], 
                color=bar_colors, height=0.7, edgecolor='white', linewidth=0.5)
        
        # Add a reference line for 100% recovery
        plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='2019 Level')
        
        plt.title('Top 15 Airports with Fastest Recovery (2024 vs. 2019)')
        plt.xlabel('Recovery Percentage (%)')
        plt.ylabel('Airport')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 5.3 Monthly patterns comparison before and after pandemic
    merge_pd['Period'] = merge_pd['YEAR'].apply(
        lambda x: 'Pre-Pandemic' if x in pre_pandemic else 
                'Pandemic' if x in pandemic else 'Post-Pandemic'
    )

    # Exclude pandemic period for clearer before/after comparison
    seasonal_df = merge_pd[merge_pd['Period'] != 'Pandemic']

    # Group by period and month
    seasonal_pattern = seasonal_df.groupby(['Period', 'MONTH_NUM'])['FLT_TOT_1'].sum().reset_index()

    # Calculate percentage of annual traffic by period
    period_totals = seasonal_pattern.groupby('Period')['FLT_TOT_1'].sum()
    seasonal_pattern['monthly_pct'] = seasonal_pattern.apply(
        lambda x: (x['FLT_TOT_1'] / period_totals[x['Period']]) * 100, axis=1
    )

    # Pivot for easier plotting
    seasonal_pivot = seasonal_pattern.pivot(index='MONTH_NUM', columns='Period', values='monthly_pct')

    plt.figure()
    # Plot with period-specific colors
    for period in seasonal_pivot.columns:
        plt.plot(seasonal_pivot.index, seasonal_pivot[period], 'o-', 
                linewidth=2.5, markersize=8, color=period_colors.get(period, '#333333'), 
                label=period)
    
    plt.title('Monthly Traffic Distribution Before and After Pandemic')
    plt.xlabel('Month')
    plt.ylabel('Percentage of Annual Traffic (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Period')
    plt.tight_layout()
    plt.show()

# ==============================================
# Question 6: COVID-19 Impact Analysis
# ==============================================

if 2019 in yearly_traffic['YEAR'].values:
    # 6.1 Pre-Pandemic Growth Analysis (2016-2019)
    pre_pandemic_trend = yearly_traffic[yearly_traffic['YEAR'].isin(pre_pandemic)]

    if len(pre_pandemic_trend) > 1:
        # Calculate compound annual growth rate (CAGR) for pre-pandemic period
        first_year = pre_pandemic_trend['FLT_TOT_1'].iloc[0]
        last_year = pre_pandemic_trend['FLT_TOT_1'].iloc[-1]
        years = len(pre_pandemic_trend) - 1
        pre_pandemic_cagr = (last_year / first_year) ** (1/years) - 1

        # 6.2 Post-Pandemic Growth Analysis (2022-2024)
        post_pandemic_trend = yearly_traffic[yearly_traffic['YEAR'].isin(post_pandemic)]
        
        if len(post_pandemic_trend) > 1:
            # Calculate CAGR for post-pandemic period
            first_year_post = post_pandemic_trend['FLT_TOT_1'].iloc[0]
            last_year_post = post_pandemic_trend['FLT_TOT_1'].iloc[-1]
            years_post = len(post_pandemic_trend) - 1
            post_pandemic_cagr = (last_year_post / first_year_post) ** (1/years_post) - 1

            # Print CAGR comparison
            print(f"Pre-pandemic CAGR (2016-2019): {pre_pandemic_cagr*100:.2f}%")
            print(f"Post-pandemic CAGR (2022-2024): {post_pandemic_cagr*100:.2f}%")

            # Calculate if 2024 levels exceeded 2019
            recovery_complete = last_year_post >= last_year
            print(f"Has air traffic fully recovered? {'Yes' if recovery_complete else 'No'}")
            print(f"2024 traffic is {last_year_post/last_year*100:.2f}% of 2019 traffic")

            # Visualization: Growth trends with pandemic impact
            plt.figure()

            # Plot actual data with enhanced styling
            scatter_sizes = [80 if year in [2019, 2020, 2024] else 60 for year in yearly_traffic['YEAR']]
            scatter_colors = [year_palette.get(int(year), '#333333') for year in yearly_traffic['YEAR']]

            plt.scatter(yearly_traffic['YEAR'], yearly_traffic['FLT_TOT_1'], 
                    s=scatter_sizes, color=scatter_colors, zorder=3,
                    edgecolor='white', linewidth=1)

            plt.plot(yearly_traffic['YEAR'], yearly_traffic['FLT_TOT_1'], '-', 
                    color='#34495E', linewidth=1.5, alpha=0.7, zorder=2, label='Actual Traffic')

            # Extend pre-pandemic trend line
            x_pre = np.array(pre_pandemic)
            y_pre = pre_pandemic_trend['FLT_TOT_1'].values
            z_pre = np.polyfit(x_pre, y_pre, 1)
            p_pre = np.poly1d(z_pre)
            x_extended = np.array(range(2016, 2025))
            y_extended = p_pre(x_extended)

            # Plot pre-pandemic trend line extended
            plt.plot(x_extended, y_extended, '--', color='#27AE60', linewidth=2, alpha=0.7, 
                    label=f'Pre-Pandemic Trend (CAGR: {pre_pandemic_cagr*100:.2f}%)')

            # Add pandemic period shading
            plt.axvspan(2020, 2021, color='#C0392B', alpha=0.15, label='Pandemic Period')

            # Add gap between actual and projected
            if y_extended[-1] > last_year_post:
                gap_pct = (y_extended[-1] - last_year_post) / y_extended[-1] * 100
                plt.annotate(f'Gap: {gap_pct:.1f}%', 
                            xy=(2024, (y_extended[-1] + last_year_post)/2),
                            xytext=(2022.5, (y_extended[-1] + last_year_post)/2),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=10)

            plt.title('European Air Traffic: Impact of COVID-19 Pandemic (2016-2024)')
            plt.xlabel('Year')
            plt.ylabel('Total Number of Flights')
            plt.grid(True, alpha=0.3)
            plt.xticks(yearly_traffic['YEAR'])
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 6.3 Country-Level Recovery Analysis
            if 2024 in years_available:
                country_2019 = merge_pd[merge_pd['YEAR'] == 2019].groupby('STATE_NAME')['FLT_TOT_1'].sum().reset_index()
                country_2024 = merge_pd[merge_pd['YEAR'] == 2024].groupby('STATE_NAME')['FLT_TOT_1'].sum().reset_index()

                # Calculate recovery percentages
                country_recovery = pd.merge(country_2019, country_2024, on='STATE_NAME', suffixes=('_2019', '_2024'))
                country_recovery['recovery_pct'] = (country_recovery['FLT_TOT_1_2024'] / country_recovery['FLT_TOT_1_2019']) * 100

                # Filter for countries with substantial traffic
                country_recovery_filtered = country_recovery[country_recovery['FLT_TOT_1_2019'] > 100000]
                country_recovery_sorted = country_recovery_filtered.sort_values('recovery_pct', ascending=False)

                plt.figure()
                # Define color mapping based on recovery percentage
                recovery_colors = [recovery_color(pct) for pct in country_recovery_sorted['recovery_pct']]

                # Create horizontal bar chart - top 15 countries
                top_countries = country_recovery_sorted.head(15)
                bars = plt.barh(y=top_countries['STATE_NAME'], width=top_countries['recovery_pct'], 
                            color=recovery_colors, height=0.7)
                
                # Add a reference line for 100% recovery
                plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='2019 Level')
                
                plt.title('Country-Level Recovery: 2024 Traffic vs. 2019 Levels')
                plt.xlabel('Recovery Percentage (%)')
                plt.ylabel('Country')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.show()

print("\nAnalysis complete. All visualizations have been generated with the custom color theme.")