# TMDb Movie Analysis - Quick Start Guide

## What's Been Done

This project has successfully replicated a comprehensive TMDb movie dataset analysis. All code and visualizations have been generated based on data analysis best practices.

## Files Created

### 1. Analysis Scripts
- **`tmdb_analysis.py`** - Complete Python script that performs full analysis
- **`tmdb_analysis.ipynb`** - Interactive Jupyter notebook for exploration

### 2. Documentation
- **`README.md`** - Comprehensive project documentation
- **`QUICKSTART.md`** - This file

### 3. Visualizations (8 PNG files)
All visualizations are automatically generated when running the analysis:
- `correlation_heatmap.png` - Shows relationships between movie features
- `budget_vs_revenue.png` - Budget vs revenue scatter plot with trend
- `top_genres.png` - Most common movie genres
- `genre_revenue.png` - Average revenue by genre
- `movies_per_year.png` - Number of movies released each year
- `runtime_distribution.png` - Distribution of movie lengths
- `top_profitable_movies.png` - Highest profit movies
- `vote_analysis.png` - Voting patterns and popularity

## How to Use

### Option 1: Run the Complete Analysis Script
```bash
cd /Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE
python3 tmdb_analysis.py
```
This will:
- Load and process the data
- Generate all 8 visualizations
- Print detailed statistical analysis
- Save all results

### Option 2: Use Jupyter Notebook (Interactive)
```bash
cd /Users/chaowang/MECHINE_LEARNING_BUSINESS/MOVIE
jupyter notebook tmdb_analysis.ipynb
```
This allows you to:
- Run analysis step-by-step
- Modify code and experiment
- View inline visualizations
- Explore data interactively

### Option 3: View Existing Visualizations
All 8 PNG files are already generated and saved. Simply open them:
```bash
open *.png
```

## Analysis Structure

### 1. Data Loading (步骤1)
- Load `tmdb_5000_movies.csv` (4,803 movies)
- Load `tmdb_5000_credits.csv` (cast and crew data)

### 2. Data Understanding (步骤2)
- Explore 20 columns in movies dataset
- Check data types and statistics
- Identify 4,803 movies from 1916-2017

### 3. Data Cleaning (步骤3)
- Merge movies and credits on ID
- Remove unnecessary columns (homepage, tagline, overview)
- Handle missing values (2 rows with missing runtime, 1 with missing release_date)
- Final dataset: 4,800 movies

### 4. Data Type Conversion (步骤4)
- Convert release_date to datetime
- Extract year and month
- Ensure proper numeric types

### 5. Data Format Conversion (步骤5)
- Parse JSON columns (genres, keywords, cast, crew, production companies)
- Extract director names
- Create aggregated features

### 6. Feature Engineering (步骤6)
- Calculate ROI: `(revenue - budget) / budget * 100`
- Calculate profit: `revenue - budget`
- Extract main genre
- Count keywords, cast members, companies

### 7. Data Visualization (步骤7)
- Generate 8 comprehensive visualizations
- All saved as high-resolution PNG files

### 8. Statistical Analysis (步骤8)
- Key metrics and insights
- Top performers
- Correlation analysis

## Key Findings

### Financial Metrics
- Average Budget: $40.65M
- Average Revenue: $121.24M
- Budget-Revenue Correlation: **0.73** (strongest predictor)

### Top 3 Movies by Revenue
1. Avatar (2009) - $2.79B
2. Titanic (1997) - $1.85B
3. The Avengers (2012) - $1.52B

### Top Directors
1. Steven Spielberg - 27 movies
2. Woody Allen - 21 movies
3. Clint Eastwood - 20 movies
4. Martin Scorsese - 20 movies

### Genre Insights
- Most Common: Drama
- Highest Revenue: Animation, Adventure

## Requirements

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

Or if using conda:
```bash
conda install pandas numpy matplotlib seaborn jupyter
```

## Troubleshooting

### Issue: Module not found
```bash
pip install pandas numpy matplotlib seaborn
```

### Issue: Jupyter not found
```bash
pip install jupyter
```

### Issue: Plot not showing
The script saves all plots as PNG files, so you can view them directly even if interactive display doesn't work.

## Next Steps

1. **Run the analysis**: `python3 tmdb_analysis.py`
2. **View visualizations**: Open the generated PNG files
3. **Explore interactively**: `jupyter notebook tmdb_analysis.ipynb`
4. **Modify and experiment**: Edit the Python script or notebook
5. **Extend analysis**: Add new visualizations or statistical tests

## Data Dictionary

### Movies Dataset Columns
- `budget` - Movie budget in USD
- `revenue` - Movie revenue in USD
- `runtime` - Duration in minutes
- `vote_average` - Average rating (0-10)
- `vote_count` - Number of votes
- `popularity` - TMDb popularity score
- `release_date` - Release date
- `genres` - JSON list of genres
- `keywords` - JSON list of keywords
- `cast` - JSON list of cast members (from credits)
- `crew` - JSON list of crew members (from credits)

### Generated Features
- `release_year` - Year extracted from release_date
- `release_month` - Month extracted from release_date
- `main_genre` - First genre from genres list
- `genres_count` - Number of genres
- `keywords_count` - Number of keywords
- `cast_count` - Number of cast members
- `director` - Director name extracted from crew
- `roi` - Return on Investment percentage
- `profit` - Revenue minus budget

## Contact & Support

For questions about the analysis or to report issues:
- Review the `README.md` for detailed documentation
- Check the Jupyter notebook for step-by-step code
- Examine the Python script for implementation details

---

**Project Status**: ✅ Complete
**Last Updated**: 2026-02-25
**Dataset**: TMDb 5000 Movies (Kaggle)
