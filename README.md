# Vendor Invoice Intelligence System

**Freight Cost Prediction & Invoice Risk Flagging**

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Business Objectives](#business-objectives)
* [Data Sources](#data-sources)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Models Used](#models-used)
* [Evaluation Metrics](#evaluation-metrics)
* [Application](#application)
* [Project Structure](#project-structure)
* [How to Run This Project](#how-to-run-this-project)
* [Author & Contact](#author--contact)

---

## 📌 Project Overview

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1.  **Predicting expected freight cost** for vendor invoices.
2.  **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

---

## 🎯 Business Objectives

### 1. Freight Cost Prediction (Regression)
**Objective:** To build a predictive model that accurately estimates the expected freight cost of incoming shipments based on historical dimensions, weight, vendor history, and route data. This helps identify overcharges before payment processing.

### 2. Invoice Risk Flagging (Anomaly Detection / Classification)
**Objective:** To develop a system that automatically scores and flags invoices exhibiting anomalous characteristics (e.g., duplicate billing, unusual price spikes, mismatched purchase orders) to minimize financial leakage and audit risks.

---

## 📊 Data Sources

* **Historical Invoice Data:** Detailed records of past vendor invoices including line items, quantities, and billed amounts.
* **Logistics & Freight Logs:** Shipping routes, carrier details, package dimensions, and fuel surcharges.
* **ERP Master Data:** Vendor profiles, standard payment terms, and historical performance metrics.

---

## 🔍 Exploratory Data Analysis

Key insights uncovered during the EDA phase included:
* **Seasonality:** Freight costs exhibit significant spikes during Q4 holiday shipping windows.
* **Outliers:** Identified specific vendor profiles consistently associated with high-variance freight billing.
* **Correlations:** Strong correlation observed between shipping distance, package weight, and fuel surcharges. 

---

## 🧠 Models Used

The intelligence system relies on a dual-model approach:
* **Regression Pipeline:** XGBoost and Random Forest Regressor for predicting continuous freight costs.
* **Anomaly Detection:** Isolation Forest and a custom threshold-based logic system to flag outliers and structural risks in invoice submissions.

---

## 📈 Evaluation Metrics

* **Regression Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).
* **Risk Flagging Metrics:** Precision, Recall, F1-Score (prioritizing high recall to ensure no high-risk invoices slip through the cracks).

---

## 💻 Application

The models are deployed via a web interface built with **Streamlit / FastAPI**. The application allows finance teams to:
* Upload bulk CSV files of new invoices.
* View predicted freight costs vs. actual billed costs.
* Download a filtered list of "High-Risk" invoices requiring immediate manual review.

---

## 📁 Project Structure

```text
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and model training
├── src/                    # Source code for data pipelines and modeling
│   ├── preprocess.py       # Data cleaning scripts
│   ├── train.py            # Model training scripts
│   └── predict.py          # Inference scripts
├── app.py                  # Web application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
