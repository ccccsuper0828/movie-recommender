"""
TMDb 5000 Movie Dataset Analysis
Based on Kaggle TMDb Movie Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Load Data
print("=" * 50)
print("1. Loading Data")
print("=" * 50)

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print(f"Movies dataset shape: {movies.shape}")
print(f"Credits dataset shape: {credits.shape}")

# 2. Understand Data
print("\n" + "=" * 50)
print("2. Understanding Data")
print("=" * 50)

print("\nMovies columns:")
print(movies.columns.tolist())

print("\nMovies info:")
print(movies.info())

print("\nMovies description:")
print(movies.describe())

print("\nCredits columns:")
print(credits.columns.tolist())

# Check if the ids match
print("\nChecking if movie ids match between datasets:")
print(f"Movies id range: {movies['id'].min()} to {movies['id'].max()}")
print(f"Credits movie_id range: {credits['movie_id'].min()} to {credits['movie_id'].max()}")

# 3. Data Cleaning
print("\n" + "=" * 50)
print("3. Data Cleaning")
print("=" * 50)

# Rename credits columns for consistency
credits.rename(columns={'movie_id': 'id'}, inplace=True)

# Check if titles match
print("\nVerifying titles match between datasets...")
merged_check = movies[['id', 'title']].merge(credits[['id', 'title']], on='id', suffixes=('_movies', '_credits'))
title_mismatch = merged_check[merged_check['title_movies'] != merged_check['title_credits']]
print(f"Title mismatches: {len(title_mismatch)}")

# Merge datasets
df = movies.merge(credits[['id', 'cast', 'crew']], on='id')
print(f"\nMerged dataset shape: {df.shape}")

# Drop unnecessary columns
columns_to_drop = ['homepage', 'tagline', 'overview']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"Dataset shape after dropping columns: {df.shape}")

# Check for missing values
print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values
print("\nHandling missing values...")
# Drop rows with missing runtime
df = df.dropna(subset=['runtime'])
# Drop rows with missing release_date
df = df.dropna(subset=['release_date'])
print(f"Dataset shape after handling missing values: {df.shape}")

# 4. Data Type Conversion
print("\n" + "=" * 50)
print("4. Data Type Conversion")
print("=" * 50)

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
print("\nConverted release_date to datetime")

# Extract year and month
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

# 5. Data Format Conversion (Parse JSON columns)
print("\n" + "=" * 50)
print("5. Data Format Conversion")
print("=" * 50)

def parse_json_column(column):
    """Parse JSON string column and extract names"""
    try:
        data = json.loads(column)
        if isinstance(data, list):
            return [item.get('name', '') for item in data if isinstance(item, dict)]
        return []
    except:
        return []

# Parse genres
print("\nParsing genres...")
df['genres_list'] = df['genres'].apply(parse_json_column)
df['genres_count'] = df['genres_list'].apply(len)
df['main_genre'] = df['genres_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')

# Parse keywords
print("Parsing keywords...")
df['keywords_list'] = df['keywords'].apply(parse_json_column)
df['keywords_count'] = df['keywords_list'].apply(len)

# Parse production companies
print("Parsing production companies...")
df['production_companies_list'] = df['production_companies'].apply(parse_json_column)
df['production_companies_count'] = df['production_companies_list'].apply(len)

# Parse cast and crew
print("Parsing cast and crew...")
df['cast_list'] = df['cast'].apply(parse_json_column)
df['cast_count'] = df['cast_list'].apply(len)

def get_director(crew_json):
    """Extract director from crew"""
    try:
        crew = json.loads(crew_json)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', 'Unknown')
        return 'Unknown'
    except:
        return 'Unknown'

df['director'] = df['crew'].apply(get_director)

# Calculate ROI (Return on Investment)
df['roi'] = ((df['revenue'] - df['budget']) / df['budget']) * 100
df['roi'] = df['roi'].replace([np.inf, -np.inf], np.nan)

# Calculate profit
df['profit'] = df['revenue'] - df['budget']

print("\nData processing complete!")
print(f"Final dataset shape: {df.shape}")

# 6. Data Visualization
print("\n" + "=" * 50)
print("6. Data Visualization")
print("=" * 50)

# Filter data for visualization (remove movies with 0 budget/revenue)
df_viz = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()

# Figure 1: Correlation Heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(12, 8))
numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
correlation_matrix = df_viz[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=1, square=True)
plt.title('Correlation Matrix of Movie Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")

# Figure 2: Budget vs Revenue
print("\nGenerating budget vs revenue scatter plot...")
plt.figure(figsize=(12, 8))
plt.scatter(df_viz['budget'] / 1e6, df_viz['revenue'] / 1e6, alpha=0.5, s=30)
plt.xlabel('Budget (Million USD)', fontsize=12)
plt.ylabel('Revenue (Million USD)', fontsize=12)
plt.title('Movie Budget vs Revenue', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(df_viz['budget'], df_viz['revenue'], 1)
p = np.poly1d(z)
plt.plot(df_viz['budget'] / 1e6, p(df_viz['budget']) / 1e6, "r--", linewidth=2, label='Trend')
plt.legend()
plt.tight_layout()
plt.savefig('budget_vs_revenue.png', dpi=300, bbox_inches='tight')
print("Saved: budget_vs_revenue.png")

# Figure 3: Top 10 Genres by Count
print("\nGenerating genre distribution...")
plt.figure(figsize=(12, 6))
genre_counts = df['main_genre'].value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Top 10 Movie Genres', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('top_genres.png', dpi=300, bbox_inches='tight')
print("Saved: top_genres.png")

# Figure 4: Average Revenue by Genre
print("\nGenerating average revenue by genre...")
plt.figure(figsize=(12, 6))
genre_revenue = df_viz.groupby('main_genre')['revenue'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=genre_revenue.values / 1e6, y=genre_revenue.index, palette='rocket')
plt.xlabel('Average Revenue (Million USD)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Top 10 Genres by Average Revenue', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('genre_revenue.png', dpi=300, bbox_inches='tight')
print("Saved: genre_revenue.png")

# Figure 5: Movies Released per Year
print("\nGenerating movies per year trend...")
plt.figure(figsize=(14, 6))
movies_per_year = df['release_year'].value_counts().sort_index()
plt.plot(movies_per_year.index, movies_per_year.values, linewidth=2, marker='o', markersize=4)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.title('Number of Movies Released per Year', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('movies_per_year.png', dpi=300, bbox_inches='tight')
print("Saved: movies_per_year.png")

# Figure 6: Runtime Distribution
print("\nGenerating runtime distribution...")
plt.figure(figsize=(12, 6))
plt.hist(df['runtime'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Runtime (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Movie Runtime', fontsize=16, fontweight='bold')
plt.axvline(df['runtime'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["runtime"].mean():.1f} min')
plt.axvline(df['runtime'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["runtime"].median():.1f} min')
plt.legend()
plt.tight_layout()
plt.savefig('runtime_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: runtime_distribution.png")

# Figure 7: Top 10 Most Profitable Movies
print("\nGenerating top profitable movies...")
plt.figure(figsize=(12, 8))
top_profit = df_viz.nlargest(10, 'profit')[['title', 'profit']].sort_values('profit')
sns.barplot(x=top_profit['profit'] / 1e9, y=top_profit['title'], palette='Greens_r')
plt.xlabel('Profit (Billion USD)', fontsize=12)
plt.ylabel('Movie Title', fontsize=12)
plt.title('Top 10 Most Profitable Movies', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('top_profitable_movies.png', dpi=300, bbox_inches='tight')
print("Saved: top_profitable_movies.png")

# Figure 8: Vote Average vs Vote Count
print("\nGenerating vote average vs count...")
plt.figure(figsize=(12, 8))
plt.scatter(df['vote_count'], df['vote_average'], alpha=0.4, s=20, c=df['popularity'], cmap='plasma')
plt.xlabel('Vote Count', fontsize=12)
plt.ylabel('Vote Average', fontsize=12)
plt.title('Vote Average vs Vote Count (colored by Popularity)', fontsize=16, fontweight='bold')
plt.colorbar(label='Popularity')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vote_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: vote_analysis.png")

# 7. Statistical Analysis Summary
print("\n" + "=" * 50)
print("7. Statistical Analysis Summary")
print("=" * 50)

print("\nKey Statistics:")
print(f"Total Movies: {len(df)}")
print(f"Date Range: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
print(f"\nAverage Budget: ${df_viz['budget'].mean() / 1e6:.2f}M")
print(f"Average Revenue: ${df_viz['revenue'].mean() / 1e6:.2f}M")
print(f"Average Runtime: {df['runtime'].mean():.1f} minutes")
print(f"Average Vote: {df['vote_average'].mean():.2f}/10")
print(f"\nMost Common Genre: {df['main_genre'].mode()[0]}")
print(f"Most Common Language: {df['original_language'].mode()[0]}")

print("\nTop 5 Highest Grossing Movies:")
top_revenue = df_viz.nlargest(5, 'revenue')[['title', 'revenue', 'release_year']]
for idx, row in top_revenue.iterrows():
    print(f"  {row['title']} ({row['release_year']:.0f}): ${row['revenue']/1e9:.2f}B")

print("\nTop 5 Highest ROI Movies:")
top_roi = df_viz.nlargest(5, 'roi')[['title', 'roi', 'budget', 'revenue']]
for idx, row in top_roi.iterrows():
    print(f"  {row['title']}: {row['roi']:.0f}% ROI (Budget: ${row['budget']/1e6:.1f}M, Revenue: ${row['revenue']/1e6:.1f}M)")

print("\nTop 5 Most Popular Directors (by movie count):")
director_counts = df['director'].value_counts().head(5)
for director, count in director_counts.items():
    if director != 'Unknown':
        print(f"  {director}: {count} movies")

print("\n" + "=" * 50)
print("Analysis Complete!")
print("=" * 50)
print("\nGenerated visualizations:")
print("  - correlation_heatmap.png")
print("  - budget_vs_revenue.png")
print("  - top_genres.png")
print("  - genre_revenue.png")
print("  - movies_per_year.png")
print("  - runtime_distribution.png")
print("  - top_profitable_movies.png")
print("  - vote_analysis.png")
