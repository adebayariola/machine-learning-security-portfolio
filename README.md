# Okeowo Adewunmi

## Machine Learning, Network Security and Data Engineering

MSc Big Data graduate with practical work in traffic classification, empirical model evaluation, data preparation and operational reporting. This portfolio presents the research question, datasets, methods, evidence and limitations for each project.

[LinkedIn](https://www.linkedin.com/in/adebayo-okeowo-87b96011b) · [GitHub](https://github.com/adebayariola)

## Start here

| Project | What it demonstrates | Evidence to explore |
|---|---|---|
| [Encrypted and Unencrypted Malware Traffic Classification](projects/security/encrypted-malware-classification/) | Four-class traffic classification with a CNN and Mamba-style block | Source code, PCAP split workflow, encryption-label logic |
| [Housing Prediction with Economic Indicators](projects/machine-learning/housing-economic-indicators/) | Housing data engineering, FRED joins and XGBoost evaluation | Notebook, five-fold metrics and figures |
| [Telecom Customer Churn Prediction](projects/machine-learning/telco-customer-churn/) | Reproducible imputation, encoding and held-out classification | Training script, saved metrics and metadata |
| [Distributed Movie Ratings Analysis](projects/data-analysis/distributed-movie-ratings/) | Equivalent movie-rating aggregation in Hadoop and Spark | Mapper, combiner, reducer and Spark implementation |

## Browse by area

[Security](#security) · [Machine learning](#machine-learning) · [Data analysis](#data-analysis) · [Systems and reporting](#systems-and-reporting)


<a id="security"></a>

## Security

| Project | Dataset / context | Publication status |
|---|---|---|
| [Network Flow Extraction and Malware Detection](projects/security/flow-malware-detection/) | USTC-TFC2016 PCAPs; CIC-IDS2017 capture inspection | Source code included |
| [Encrypted and Unencrypted Malware Traffic Classification](projects/security/encrypted-malware-classification/) | Labelled PCAP manifests; mixed benign and malware flows | Source code included |
| [Two-Stage Confidence-Gated Malware Detection](projects/security/two-stage-malware-detection/) | USTC-TFC2016 benign and malware traffic | Code in related flow-detection project |
| [Open-Set Traffic Classification and CNN State-Mixing Variants](projects/security/open-set-traffic-classification/) | TrafficGPT-format USTC and ISCX data; configurable DF, AWF, DC and CSTNet preparation | Source code included |
| [HTTPS Traffic Classification Pipelines](projects/security/https-traffic-classification/) | HTTPS-clf-dataset.csv; Kaggle HTTPS Traffic Classification | Source code included |
| [Self-Supervised Malware Traffic Classification](projects/security/self-supervised-traffic-classification/) | USTC-TFC2016; ISCX VPN-NonVPN | Project summary; source not published |
| [ETGuard Sequence Classification and Alternative Baseline](projects/security/etguard-sequence-classification/) | USTC-TFC2016 and DoHBrw PCAP adapters | Project summary; source not published |
| [IoT Malware Traffic Data Preparation](projects/security/iot23-data-preparation/) | IoT-23 labelled network traffic | Project summary; source not published |

<a id="machine-learning"></a>

## Machine learning

| Project | Dataset / context | Publication status |
|---|---|---|
| [Residential Property Price Prediction](projects/machine-learning/housing-dissertation/) | King County Sales by Andy Krause, Kaggle; kingCoData | Project summary; source not published |
| [Housing Prediction with Economic Indicators](projects/machine-learning/housing-economic-indicators/) | King County Sales; FRED MORTGAGE30US, CUUSA423SA0, SEAT653URN and PCPI53033 | Code and saved evaluation included |
| [Telecom Customer Churn Prediction](projects/machine-learning/telco-customer-churn/) | IBM Telco Customer Churn; WA_Fn-UseC_-Telco-Customer-Churn.csv | Code and saved evaluation included |
| [Telecommunications Campaign Response Classification](projects/machine-learning/wallace-campaign-response/) | Wallace Communications; wallacecommunications.csv; 50,662 customer records | Project summary; source not published |

<a id="data-analysis"></a>

## Data analysis

| Project | Dataset / context | Publication status |
|---|---|---|
| [Retail Product and Customer Review Analysis](projects/data-analysis/retail-review-analysis/) | JC Penney products.csv, reviews.csv, users.csv, jcpenney_products.json and jcpenney_reviewers.json | Project summary; source not published |
| [Distributed Movie Ratings Analysis](projects/data-analysis/distributed-movie-ratings/) | Course ratings.txt, years.txt and genres.txt | Source code included |
| [Football Match Statistics Portfolio](projects/data-analysis/football-statistics/) | Manchester United 2023/24 Premier League match records; 30 matches | Project summary; source not published |
| [Hotel Room Demand Forecasting](projects/data-analysis/hotel-demand-forecasting/) | Course-generated monthly hotel-demand series; 100 observations | Project summary; source not published |
| [Customer Churn Score Modelling](projects/data-analysis/churn-score-regression/) | Course-generated customer attributes and continuous churn scores; 1,000 customers | Project summary; source not published |
| [Cybersecurity Event Forecasting](projects/data-analysis/cyber-event-forecasting/) | Course-generated monthly phishing, malware, suspicious URL and spam indicators | Project summary; source not published |

<a id="systems-and-reporting"></a>

## Systems and reporting

| Project | Dataset / context | Publication status |
|---|---|---|
| [Sound Festival Schedule Builder](projects/systems-and-reporting/festival-schedule-builder/) | Linked festival events, artists, venues, staff, bookings, repertoire and supplier records | Project summary; source not published |
| [Student Lifestyle Dashboard](projects/systems-and-reporting/student-lifestyle-dashboard/) | Student lifestyle dataset; 2,000 records | Project summary; source not published |
| [Sales Analysis Dashboards](projects/systems-and-reporting/sales-dashboards/) | Sales coursework records | Project summary; source not published |

## Reading the evidence

**Code included** means source is available to inspect. **Saved evaluation included** means the repository also contains retained metrics or figures. **Project summary** means the work is described but its source is not published here. These labels describe publication contents, not a claim of independent reproduction.

Security draft scores are withheld pending verification. The four-class network model uses a lightweight Mamba-style block, and encryption status is inferred from traffic signatures. The generated-data forecasting coursework is identified explicitly and kept separate from the IBM Telco and Wallace classification projects.

## Repository layout

```text
projects/
  security/                 # Network traffic and malware research
  machine-learning/         # Housing, IBM Telco and Wallace projects
  data-analysis/            # Retail, Hadoop/Spark, statistics and forecasting
  systems-and-reporting/    # Festival scheduling and dashboards
docs/                       # Portfolio guide and evidence policy
```

Each project has its own `README.md`. Published source keeps its working directory structure, including `src/`, dependency files, notebooks and results where available. Run project commands from the corresponding project folder.

## Data and attribution

Raw data, model binaries and private charity records are excluded. Obtain datasets from their owners and follow their terms. Original upstream architectures and paper results are distinguished from local adaptations. See the [portfolio guide](docs/PORTFOLIO_GUIDE.md).
