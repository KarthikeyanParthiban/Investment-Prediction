# Value Spectrum: Investment prediction 

## Objective

To increase the average transaction value (ATV) by leveraging user behavior insights and introducing a "Value Spectrum"—a dynamic comparison tool that highlights the impact of incremental investment adjustments on long-term returns.

## Problem Breakdown

- Over 40% of users modify their investment amount before completing a payment.

- These changes happen without any business intervention, indicating a behavioral opportunity.

- The challenge is to introduce subtle yet impactful nudges that enhance decision-making.

## Solution: The Value Spectrum Approach

  ### 1. Behavioral Insight & Data Analysis

 - Analyzed historical transaction data, focusing on:

 - Initial vs. Final Investment Amount

 - User modification patterns (+/-)

 - Interest earned on different investment values

 - Payment behavior & user demographics

  ### 2. Predictive Modeling: XGBoost for Investment Predictions

  Built an XGBoost regression model to predict a user’s final investment amount.

   ### Performance Metrics:
    
    R²: 0.93 (High Accuracy)
    
    MAE: ₹45,065.47
    
    RMSE: ₹1,55,247.57
  
### 3. Designing the Value Spectrum Nudge

  The Value Spectrum is a dynamic comparison tool that presents:

 -  Initial Investment Amount & Expected Interest Earned
  
 -  Suggested Investment Amount & Enhanced Interest Earned
  
 -  A visual representation of long-term gains to encourage better decision-making


### Measured impact on:

  Average Transaction Value (ATV)
  
  Conversion Rates
  
  User Engagement


### Implementation Strategy

#### Example : Pre-Payment Page (Before Confirmation)

  #### Why?
  
   - Users haven’t finalized their investment amount.
    
   - They are open to reconsidering based on added benefits.
  
  ####  Implementation Idea:
    
   - Show a comparison table of current vs. suggested investment.
    
   - Use a slider to let users see the impact of different amounts.



  
