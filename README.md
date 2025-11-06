# ClimateRisk360 – ESG & Insurance Risk Data Integration Platform

## 📝 Project Overview
ClimateRisk360 is a data integration and analytics platform that combines climate data with insurance claims to assess and visualize climate-related risks for insurance companies. The platform provides actionable insights through a user-friendly dashboard.

## 🚀 Features
- **Data Integration**: Combines climate and insurance data from multiple sources
- **Risk Analysis**: Calculates climate risk scores for different regions
- **Interactive Dashboard**: Visualizes risk heatmaps, claim trends, and climate correlations
- **Modular Pipeline**: Built with PySpark for scalable data processing

## 🛠️ Tech Stack
- **Backend**: Python, PySpark, Pandas
- **Visualization**: Streamlit, Plotly
- **Data Storage**: Local file system (CSV/Parquet)
- **Development**: Git, Pytest, Black

## 🏗️ Project Structure
```
ClimateRisk360/
├── data/                    # Raw and processed data
├── src/                     # Source code
│   ├── data_ingestion.py    # Data loading utilities
│   ├── data_cleaning.py     # Data cleaning functions
│   ├── data_processing.py   # Data transformation logic
│   ├── analytics.py         # Risk score calculation
│   └── visualization_app.py # Streamlit dashboard components
├── streamlit_app.py         # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Java 8 or later (for PySpark)

### Installation
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Start the Streamlit dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your browser to `http://localhost:8501`

## 📊 Data Model

### Key Datasets
- **Insurance Claims**: Policy details, claim amounts, dates, and locations
- **Climate Data**: Temperature, precipitation, and extreme weather events
- **Region Mapping**: Geographic boundaries and region codes

### Calculated Metrics
- Climate Risk Score
- Claim Frequency
- Average Claim Amount
- Climate Event Impact

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
Your Name - [@atharva_ahire](https://www.linkedin.com/in/atharva-ahire-9b2796303/) - atharvaahire07@gmail.com

## 🙏 Acknowledgments
- Public climate data providers
- Open-source community for amazing libraries
- Insurance industry research papers
