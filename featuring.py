# For CatBoost (our best model)
if best_model_name == 'CatBoost':
    # Get feature names after one-hot encoding
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numeric_features + list(cat_features)
    
    # Get feature importance
    importance = best_model.named_steps['regressor'].get_feature_importance()
    feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importance})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20))
    plt.title('Top 20 Feature Importance')
    plt.show()