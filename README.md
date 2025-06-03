 Predicting-Customer-Review-Scores-for-BLU-Big-Data-Project- A Spark-powered machine learning solution to boost e-commerce customer satisfaction

 Project Overview
Objective: Predict whether BLU’s customers will leave positive (4-5) or negative (1-3) reviews based on order features (delivery performance, product details, payment behavior).
Business Impact: Address BLU’s repeat customer gap (6.2% vs. competitors’ 14-19%) by improving review scores and loyalty.

 Key Results
Best Model: Gradient Boosting achieved 81.6% accuracy and AUC 0.706.

Top Drivers of Positive Reviews:

Timely deliveries (reducing delivery_delay and customer_wait_time).

Detailed product descriptions (longer descriptions correlated with higher scores).

Larger orders (order_item_count positively impacted ratings).

Business Recommendations:

Optimize logistics to minimize delays.

Incentivize multi-item purchases (e.g., bundle discounts).

Enhance product pages with richer descriptions.

 Technical Approach
Data & Tools
Datasets: 50K+ orders (2020–2022) with product, payment, and review data.

Tech Stack: PySpark for big data processing, scikit-learn for modeling.

Methodology
Feature Engineering:

Created key variables like delivery_delay, customer_wait_time, and heavy_order (weight > 10kg).

Aggregated product metadata (e.g., average description length).

Model Selection: Compared Logistic Regression, Random Forest, and Gradient Boosting (best performance).

Evaluation: Prioritized accuracy and AUC, validated via ROC curves and confusion matrices.

 Repository Structure
├── data/                # Processed datasets (sample)  
├── notebooks/           # PySpark code for ETL, modeling, and evaluation  
├── presentation/        # Slides   
├── README.md            # This overview  
└── requirements.txt     # Python dependencies  
 
 Why This Matters
BLU can increase repeat business by focusing on the levers identified (e.g., faster deliveries, better product info). Our model provides a scalable framework to predict and improve customer satisfaction.

 #BigData #PySpark #CustomerAnalytics #GradientBoosting #Ecommerce
