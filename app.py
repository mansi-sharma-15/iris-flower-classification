
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, auc,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# ==========================================================
# IRIS MULTI-CLASS CLASSIFICATION - INTERACTIVE DASHBOARD
# ==========================================================

st.set_page_config(
    page_title="Iris ML Analytics Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_package():
    return joblib.load("iris_best_model.pkl")

package = load_package()
model = package["model"]
scaler = package["scaler"]
pca = package["pca"]
lda = package["lda"]
nb_binner = package.get("nb_binner")

class_names = package["class_names"]
class_labels = np.asarray(package["class_labels"])
feature_names = package["feature_names"]

y_true = np.asarray(package["y_test"])
y_pred = np.asarray(package["y_pred"])
y_prob = np.asarray(package["y_prob"])
cv_summary = package.get("cv_summary", {})
cv_folds = int(package.get("cv_folds", 5))

# Align probability columns with the model's class order.
label_to_name = {int(label): name for label, name in zip(class_labels, class_names)}
ordered_names = [label_to_name[int(label)] for label in model.classes_]

# Species photographs hosted on Wikimedia Commons (shown with attribution).
# The images are illustrative species references, not photographs of the input sample.
SPECIES_PHOTOS = {
    "setosa": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Wild_iris_flower_iris_setosa.jpg",
        "credit": "U.S. Fish and Wildlife Service · Public domain",
        "source": "https://commons.wikimedia.org/wiki/File:Wild_iris_flower_iris_setosa.jpg",
    },
    "versicolor": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor_FWS.jpg",
        "credit": "U.S. Fish and Wildlife Service · Public domain",
        "source": "https://commons.wikimedia.org/wiki/File:Iris_versicolor_FWS.jpg",
    },
    "virginica": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg",
        "credit": "Frank Mayfield · CC BY-SA 2.0",
        "source": "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg",
    },
}

# Recalculate dashboard metrics from the saved held-out test predictions.
cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
report = classification_report(
    y_true, y_pred, labels=model.classes_,
    target_names=ordered_names, output_dict=True, zero_division=0
)
acc = accuracy_score(y_true, y_pred)
p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
r_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
f_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
bal_acc = balanced_accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
try:
    auc_weighted = roc_auc_score(
        y_true, y_prob, labels=model.classes_,
        multi_class="ovr", average="weighted"
    )
    auc_macro = roc_auc_score(
        y_true, y_prob, labels=model.classes_,
        multi_class="ovr", average="macro"
    )
    logloss = log_loss(y_true, y_prob, labels=model.classes_)
except Exception:
    auc_weighted, auc_macro, logloss = np.nan, np.nan, np.nan

# One-vs-rest specificity and negative predictive value for each class.
specificity_by_class = {}
npv_by_class = {}
for i, label in enumerate(model.classes_):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    specificity_by_class[ordered_names[i]] = tn / (tn + fp) if (tn + fp) else 0.0
    npv_by_class[ordered_names[i]] = tn / (tn + fn) if (tn + fn) else 0.0

macro_specificity = float(np.mean(list(specificity_by_class.values())))
macro_npv = float(np.mean(list(npv_by_class.values())))
try:
    y_onehot = label_binarize(y_true, classes=model.classes_)
    macro_ap = average_precision_score(y_onehot, y_prob, average="macro")
    multiclass_brier = float(np.mean(np.sum((y_onehot - y_prob) ** 2, axis=1)))
except Exception:
    macro_ap = np.nan
    multiclass_brier = np.nan

# Sidebar controls
st.sidebar.title("🌸 Dashboard Controls")
st.sidebar.caption("Explore model evaluation and test new flower measurements.")
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Detailed Metrics", "Confusion Matrix",
     "ROC & Precision-Recall", "Model Comparison",
     "Try a Prediction", "About the Model"]
)
st.sidebar.divider()
st.sidebar.write("**Selected model**")
st.sidebar.write(package["model_name"])
st.sidebar.write("**Feature representation**")
st.sidebar.write(package["feature_representation"])

st.title("🌸 Iris Multi-Class Classification")
st.markdown(
    "An interactive analytics dashboard for evaluating and exploring "
    "a machine-learning classifier for Setosa, Versicolor and Virginica."
)
st.caption(
    "Evaluation metrics below are calculated from the notebook's held-out test set. "
    "Prediction inputs are separate from model evaluation."
)

# Reusable KPI row
def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)

if section == "Overview":
    st.subheader("Model Performance at a Glance")
    st.caption(
        f"Headline scores are the mean of {cv_folds}-fold stratified cross-validation "
        "on the training partition. The held-out test set is kept separate."
    )
    metric_row([
        ("CV Accuracy", f"{cv_summary.get('CV Accuracy Mean (%)', np.nan):.2f}%",
         "Mean accuracy across training-only cross-validation folds."),
        ("CV Precision · weighted", f"{cv_summary.get('CV Precision (Weighted) (%)', np.nan):.2f}%",
         "Mean weighted precision across cross-validation folds."),
        ("CV Recall · weighted", f"{cv_summary.get('CV Recall (Weighted) (%)', np.nan):.2f}%",
         "Mean weighted recall across cross-validation folds."),
        ("CV F1 · weighted", f"{cv_summary.get('CV F1-Score (Weighted) (%)', np.nan):.2f}%",
         "Mean weighted F1 across cross-validation folds.")
    ])
    cv_std = cv_summary.get("CV Accuracy Std (%)", np.nan)
    st.metric(
        "Cross-validation accuracy variation",
        f"±{cv_std:.2f} percentage points",
        help="Standard deviation of accuracy across the five training folds."
    )
    st.info(
        "The final held-out test-set scores are shown under **Detailed Metrics**. "
        "They are not used to choose the model."
    )
    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Class distribution in the test set")
        counts = pd.Series(y_true).map(label_to_name).value_counts().reindex(ordered_names)
        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Species", "y": "Test observations"},
            text=counts.values, color=counts.index,
            title="Test-set observations by species"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Correct vs incorrect predictions")
        correct = int(np.sum(y_true == y_pred))
        incorrect = int(len(y_true) - correct)
        fig = px.pie(
            names=["Correct", "Incorrect"], values=[correct, incorrect],
            hole=0.58, title="Prediction outcome"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"Model: **{package['model_name']}** · "
        f"Features: **{package['feature_representation']}** · "
        f"Test samples: **{len(y_true)}**"
    )

elif section == "Detailed Metrics":
    st.subheader("Held-out Test Set Evaluation Metrics")
    st.caption("These are actual calculated scores on the untouched 20% test split.")
    overall = pd.DataFrame([
        ["Accuracy", acc],
        ["Precision (macro)", p_macro],
        ["Precision (weighted)", p_weighted],
        ["Recall (macro)", r_macro],
        ["Recall (weighted)", r_weighted],
        ["F1-score (macro)", f_macro],
        ["F1-score (weighted)", f_weighted],
        ["Balanced accuracy", bal_acc],
        ["Specificity (macro)", macro_specificity],
        ["Negative predictive value (macro)", macro_npv],
        ["ROC-AUC (macro, OvR)", auc_macro],
        ["ROC-AUC (weighted, OvR)", auc_weighted],
        ["Macro average precision (OvR)", macro_ap],
        ["Matthews correlation coefficient", mcc],
        ["Cohen's kappa", kappa],
        ["Log loss", logloss],
        ["Multiclass Brier score", multiclass_brier]
    ], columns=["Metric", "Score"])
    st.dataframe(
        overall.style.format({"Score": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "For single-label multiclass classification, weighted recall equals accuracy. "
        "Macro averages give each species equal importance; weighted averages account "
        "for class support."
    )
    st.markdown("#### Training-only cross-validation summary")
    cv_table = pd.DataFrame([{
        "Folds": cv_folds,
        "CV Accuracy Mean (%)": cv_summary.get("CV Accuracy Mean (%)", np.nan),
        "CV Accuracy Std (%)": cv_summary.get("CV Accuracy Std (%)", np.nan),
        "CV Precision (Weighted) (%)": cv_summary.get("CV Precision (Weighted) (%)", np.nan),
        "CV Recall (Weighted) (%)": cv_summary.get("CV Recall (Weighted) (%)", np.nan),
        "CV F1-Score (Weighted) (%)": cv_summary.get("CV F1-Score (Weighted) (%)", np.nan)
    }])
    st.dataframe(cv_table.round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Per-species metrics")
    rows = []
    for name in ordered_names:
        item = report[name]
        rows.append({
            "Species": name,
            "Precision": item["precision"],
            "Recall / Sensitivity": item["recall"],
            "F1-score": item["f1-score"],
            "Specificity": specificity_by_class[name],
            "NPV": npv_by_class[name],
            "Support": int(item["support"])
        })
    per_class = pd.DataFrame(rows)
    st.dataframe(
        per_class.style.format({
            "Precision": "{:.3f}", "Recall / Sensitivity": "{:.3f}",
            "F1-score": "{:.3f}", "Specificity": "{:.3f}", "NPV": "{:.3f}"
        }),
        use_container_width=True, hide_index=True
    )
    metric_choice = st.multiselect(
        "Choose metrics to compare by species",
        ["Precision", "Recall / Sensitivity", "F1-score", "Specificity", "NPV"],
        default=["Precision", "Recall / Sensitivity", "F1-score"]
    )
    if metric_choice:
        melted = per_class.melt(
            id_vars="Species", value_vars=metric_choice,
            var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            melted, x="Species", y="Score", color="Metric",
            barmode="group", range_y=[0, 1.05],
            title="Per-species metric comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "⬇️ Download evaluation metrics (CSV)",
        overall.to_csv(index=False).encode("utf-8"),
        "iris_overall_metrics.csv", "text/csv"
    )
    st.download_button(
        "⬇️ Download per-species report (CSV)",
        per_class.to_csv(index=False).encode("utf-8"),
        "iris_classification_report.csv", "text/csv"
    )

elif section == "Confusion Matrix":
    st.subheader("Interactive Confusion Matrix")
    normalize = st.selectbox(
        "Display values as",
        ["Raw counts", "Normalize by actual class (recall)", "Normalize by predicted class (precision)"]
    )
    shown = cm.astype(float)
    if normalize.startswith("Normalize by actual"):
        denom = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, denom, out=np.zeros_like(shown), where=denom != 0)
    elif normalize.startswith("Normalize by predicted"):
        denom = shown.sum(axis=0, keepdims=True)
        shown = np.divide(shown, denom, out=np.zeros_like(shown), where=denom != 0)
    text_values = np.round(shown, 3) if normalize != "Raw counts" else cm
    fig = go.Figure(data=go.Heatmap(
        z=shown, x=ordered_names, y=ordered_names,
        text=text_values, texttemplate="%{text}",
        colorscale="Blues", colorbar_title="Score"
    ))
    fig.update_layout(
        title=normalize, xaxis_title="Predicted species",
        yaxis_title="Actual species"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rows represent actual classes; columns represent predicted classes.")
    st.markdown("#### Misclassification inspection")
    errors = pd.DataFrame({
        "Actual": [label_to_name[int(v)] for v in y_true],
        "Predicted": [label_to_name[int(v)] for v in y_pred],
        "Correct": y_true == y_pred,
        "Confidence": np.max(y_prob, axis=1)
    })
    only_errors = st.checkbox("Show only misclassified test observations", value=True)
    view = errors.loc[~errors["Correct"]] if only_errors else errors
    st.dataframe(view.reset_index(names="Test row"), use_container_width=True, hide_index=True)
    st.metric("Misclassified observations", int((~errors["Correct"]).sum()))

elif section == "ROC & Precision-Recall":
    st.subheader("One-vs-Rest ROC Curves")
    y_bin = label_binarize(y_true, classes=model.classes_)
    roc_fig = go.Figure()
    for i, name in enumerate(ordered_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={class_auc:.3f})"
        ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random baseline", line=dict(dash="dash")
    ))
    roc_fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        title="Multiclass ROC curves (one-vs-rest)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02])
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    st.subheader("Precision-Recall Curves")
    pr_fig = go.Figure()
    for i, name in enumerate(ordered_names):
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_bin[:, i], y_prob[:, i]
        )
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        pr_fig.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve, mode="lines",
            name=f"{name} (AP={ap:.3f})"
        ))
    pr_fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        title="Precision-Recall curves (one-vs-rest)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02])
    )
    st.plotly_chart(pr_fig, use_container_width=True)
    st.subheader("Probability calibration")
    st.caption("A reliability curve compares predicted probabilities with observed class frequencies.")
    calibration_fig = go.Figure()
    for i, name in enumerate(ordered_names):
        observed_fraction, mean_predicted = calibration_curve(
            y_bin[:, i], y_prob[:, i], n_bins=5, strategy="uniform"
        )
        calibration_fig.add_trace(go.Scatter(
            x=mean_predicted, y=observed_fraction, mode="lines+markers",
            name=name
        ))
    calibration_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect calibration", line=dict(dash="dash")
    ))
    calibration_fig.update_layout(
        title="Class-wise calibration curves",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive frequency",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(calibration_fig, use_container_width=True)
    st.caption(
        "ROC and precision-recall curves use predicted class probabilities "
        "from the held-out test set. Calibration curves are descriptive; "
        "the Iris test set is small, so interpret them cautiously."
    )

elif section == "Model Comparison":
    st.subheader("Compare all trained model / feature combinations")
    comparison = pd.DataFrame(package.get("model_comparison", []))
    if not comparison.empty:
        comparison["Testing Accuracy (%)"] = pd.to_numeric(
            comparison["Testing Accuracy (%)"], errors="coerce"
        )
        pivot = comparison.pivot(
            index="ML Model", columns="Feature Representation",
            values="Testing Accuracy (%)"
        )
        st.dataframe(pivot.style.format("{:.2f}%"), use_container_width=True)
        available_metrics = [
            col for col in comparison.columns
            if col not in ["ML Model", "Feature Representation"]
        ]
        selected_metric = st.selectbox(
            "Choose a metric for model comparison",
            available_metrics,
            index=0
        )
        fig = px.bar(
            comparison, x="ML Model", y=selected_metric,
            color="Feature Representation", barmode="group",
            title=f"{selected_metric} across all model configurations"
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "⬇️ Download model comparison (CSV)",
            comparison.to_csv(index=False).encode("utf-8"),
            "iris_model_comparison.csv", "text/csv"
        )
    else:
        st.warning("Model comparison data was not found in the saved package.")

elif section == "Try a Prediction":
    st.subheader("Enter flower measurements")
    st.write("Adjust the four measurements, then select **Predict species**.")
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
    with c2:
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names
    )
    st.markdown("**Current input measurements**")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    if st.button("🌼 Predict Iris species", type="primary"):
        scaled = scaler.transform(input_df)
        representation = package["feature_representation"]
        if representation == "Original Features":
            transformed = scaled
        elif representation == "PCA Features":
            transformed = pca.transform(scaled)
        elif representation == "LDA Features":
            transformed = lda.transform(scaled)
        else:
            st.error("Unknown feature representation in saved model.")
            st.stop()

        if package["model_name"] == "Naive Bayes":
            if nb_binner is None:
                st.error("Naive Bayes discretizer is missing from the model package.")
                st.stop()
            transformed = nb_binner.transform(transformed).astype(int)

        prediction = model.predict(transformed)[0]
        probabilities = model.predict_proba(transformed)[0]
        predicted_name = label_to_name[int(prediction)]
        confidence = float(np.max(probabilities))

        st.success(f"Predicted species: **{predicted_name}**")
        st.metric("Prediction confidence", f"{confidence:.1%}")

        # Show the matching reference photograph for the predicted species.
        photo = SPECIES_PHOTOS.get(predicted_name.strip().lower())
        if photo:
            st.markdown(f"#### 🌸 {predicted_name} flower")
            st.image(photo["url"], caption=f"{predicted_name} · {photo['credit']}", width=420)
            st.caption(f"Photo source: {photo['source']}")
        else:
            st.info("A reference photograph is not configured for this species.")
        probability_df = pd.DataFrame({
            "Species": ordered_names,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        prob_fig = px.bar(
            probability_df, x="Species", y="Probability",
            color="Species", range_y=[0, 1],
            title="Predicted probability by species",
            text=probability_df["Probability"].map(lambda x: f"{x:.1%}")
        )
        st.plotly_chart(prob_fig, use_container_width=True)
        st.dataframe(
            probability_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True, hide_index=True
        )
        st.caption(
            "Confidence is the highest predicted class probability; it is not "
            "a guarantee that the prediction is correct."
        )

elif section == "About the Model":
    st.subheader("Model and experiment information")
    st.write(f"**Algorithm:** {package['model_name']}")
    st.write(f"**Feature representation:** {package['feature_representation']}")
    st.write(f"**Input measurements:** {', '.join(feature_names)}")
    st.write(f"**Species:** {', '.join(class_names)}")
    st.write(f"**Held-out test observations:** {len(y_true)}")
    st.write("**Evaluation protocol:** Metrics are computed on the existing stratified test split.")
    st.warning(
        "The dashboard reports test-set performance for the selected model. "
        "Do not interpret interactive user predictions as additional test-set evaluation."
    )

st.divider()
st.caption(
    "Iris Multi-Class Classification Project · Streamlit · "
    "Evaluation based on the notebook's held-out test data"
)
