# Credit Risk Assessment and Portfolio Optimization Project Using SAS

> This repository documents the project's methodology, data sources, and SAS toolset — it's a design/reference write-up rather than a repository of runnable SAS scripts.

## Project Overview
This project demonstrates the application of advanced statistical methods using SAS software to analyze credit risk and optimize investment portfolios for a mid-sized financial institution. The analysis combines traditional statistical techniques with machine learning approaches to develop a comprehensive credit scoring model and portfolio optimization strategy.

## Data Sources
**Primary data sources:**
- Loan-level data from the institution's core banking system (10 years of historical data)
- Credit bureau reports (TransUnion, Equifax)
- Macroeconomic indicators from Federal Reserve Economic Data (FRED)
- Market data from Bloomberg terminal exports
- Basel III regulatory reporting data

**Data dimensions:**
- 2.5 million loan records
- 200+ potential predictor variables
- Time series data spanning quarterly observations from 2015-2024

## SAS Tools and Packages Utilized

### Data Preparation
- **SAS Base**: data cleaning, transformation, and standardization
- **SAS/ACCESS**: connection to Oracle and SQL Server databases
- **SAS Data Integration Studio**: data pipelines and ETL workflows
- **PROC IMPORT/EXPORT**: handling external file formats (CSV, Excel)
- **PROC SQL**: complex data joins and aggregations

### Statistical Analysis
- **SAS/STAT**: core statistical procedures
- **PROC FREQ, PROC MEANS, PROC UNIVARIATE**: descriptive statistics
- **PROC CORR, PROC REG**: correlation and regression analysis
- **PROC FACTOR, PROC PRINCOMP**: factor analysis and principal component analysis
- **PROC DISCRIM, PROC LOGISTIC**: classification models
- **PROC GLIMMIX**: generalized linear mixed models
- **PROC VARCLUS**: variable clustering

### Advanced Analytics
- **SAS Enterprise Miner**: predictive modeling workflow
- **PROC HPFOREST**: high-performance random forests
- **SAS Visual Analytics**: interactive visualization and exploration
- **PROC OPTMODEL**: portfolio optimization
- **SAS/ETS**: time series analysis and forecasting
- **PROC ARIMA, PROC AUTOREG**: time series modeling
- **PROC PANEL**: panel data analysis

### Reporting and Visualization
- **SAS ODS Graphics**: custom visualization development
- **PROC SGPLOT, PROC SGPANEL**: statistical graphics
- **SAS Visual Analytics**: dashboard creation and interactive reports

## Project Methodology
1. **Data Preparation and Exploratory Analysis**
2. **Feature Engineering and Variable Selection**
3. **Credit Risk Modeling**
4. **Portfolio Optimization**
5. **Stress Testing and Scenario Analysis**
6. **Regulatory Capital Calculation**
7. **Reporting and Visualization**

## Future Enhancements
- Integration of alternative data sources for enhanced predictive power
- Implementation of machine learning techniques (gradient boosting, neural networks)
- Development of a real-time scoring API for instant credit decisions
- Creation of interactive risk dashboards using SAS Visual Analytics
