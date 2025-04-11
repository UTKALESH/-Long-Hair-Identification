# Conditional Gender Identification from Images

This project implements a machine learning system to predict a person's gender from an image based on a specific set of conditional rules involving age and hair length.

## Project Goal

The primary goal is to build a system that adheres to the following logic:

1.  Predict the person's **Age Group** from the input image (<20, 20-30, >30).
2.  **If the predicted Age Group is 20-30:**
    *   Predict the person's **Hair Length** (Long/Short).
    *   If Hair Length is Long, the **Final Output Gender** is **Female**.
    *   If Hair Length is Short, the **Final Output Gender** is **Male**.
    *   *(The actual gender is ignored in this age bracket)*.
3.  **If the predicted Age Group is < 20 OR > 30:**
    *   Predict the person's actual **Gender** (Male/Female).
    *   The **Final Output Gender** is the **predicted actual gender**.
    *   *(Hair length is ignored in these age brackets)*.

The project involves training separate deep learning models for Age Group, Gender, and Hair Length classification and implementing the conditional logic in a user-friendly GUI.






