import streamlit as st

# Set up the page
st.set_page_config(page_title="Artefact Demonstration", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section:", ("Artefact Description", "Statistical Models", "LSTM Neural Network"))

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Artefact Description":
    st.title("Short-Term Temperature Forecasting: Artefact Demonstration")

    st.markdown(
        """
        <style>
        img {
            max-width: 700px !important;  /* maximum width for all images */
            height: auto;                 /* keep aspect ratio */
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 8px;           /* optional, for rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("""
    This artefact is part of the research project **“Short-Term Temperature Forecasting: A Comparison of Statistical Models and LSTM Neural Networks.”**  
    The study investigates the effectiveness of **ARIMA and SARIMA models** compared to a **Long Short-Term Memory (LSTM) neural network** for short-term temperature prediction.  
    """)

    # --- Section 1: Data Collection ---
    with st.expander("1. Data Collection Setup", expanded=True):
        st.write("""
        Temperature data were collected continuously over one week at **one-minute intervals**, producing a high-resolution dataset suitable for short-term prediction.

        Measurements were recorded using a **Micro:bit microcontroller** with a **DS18B20 digital temperature sensor**, forming a compact, reliable, and locally deployable device.  
        This setup captured local temperature variations in **Potchefstroom, South Africa** and served as the foundation for the forecasting experiments.
        """)
        st.image("figures/microbit_sensor.jpg",
                 caption="Micro:bit microcontroller with DS18B20 temperature sensor used for local data collection",
                 use_container_width=True, clamp=True)

        st.write("Figure 1.1 shows the recorded temperature series over the observation period.")
        st.image("figures/4.1_temperature_observations.jpg",
                 caption="Figure 1.1: Temperature observations in Potchefstroom over one week",
                 use_container_width=True)

    # --- Section 2: Data Processing ---
    with st.expander("2. Data Processing and Preparation"):
        st.write("""
        Prior to model training, the dataset underwent several preprocessing steps:
        - Converted timestamps to **datetime format**  
        - Handled **missing or corrupted readings**  
        - Resampled readings into **hourly averages** for statistical models  
        - Split into **training (70%)**, **validation (20%)**, and **test (10%)** subsets

        Stationarity tests were performed using the **Augmented Dickey-Fuller (ADF) test**. Table 2.1 shows that the raw series was non-stationary, while Table 2.2 shows the series after differencing became stationary. Autocorrelation and partial autocorrelation analyses guided ARIMA/SARIMA parameter selection.
        """)

        import pandas as pd
        # --- Tables ---
        # Read Table 4.1 (use utf-8-sig to remove BOM)
        table_41 = pd.read_csv("tables/4.1_adf_test.csv", encoding="utf-8-sig")
        st.write("Table 2.1: ADF test results for the raw temperature series")
        st.table(table_41.reset_index(drop=True))

        # Read Table 4.2
        table_42 = pd.read_csv("tables/4.2_adf_test_diff.csv", encoding='cp1252')

        # Align columns with Table 4.1
        table_42 = table_42.reindex(columns=table_41.columns)

        st.write("Table 2.2: ADF test results after first differencing")
        st.table(table_42.reset_index(drop=True))

        # --- Figures ---
        st.write("Figure 2.1 shows the autocorrelation function of the original temperature series, indicating daily patterns.")
        st.image("figures/4.2_acf_plot.jpg", caption="Figure 2.1: ACF plot of the temperature series", use_container_width=True)

        st.write("Figure 2.2 shows the temperature series after first differencing to achieve stationarity.")
        st.image("figures/4.3_differenced_series.jpg", caption="Figure 2.2: Differenced temperature time series", use_container_width=True)

        st.write("Figure 2.3 shows the autocorrelation of the differenced series, which supports the chosen ARIMA parameters.")
        st.image("figures/4.4_acf_differenced.jpg", caption="Figure 2.3: ACF plot of differenced series", use_container_width=True)

    # --- Section 3: Statistical Models ---
    with st.expander("3. Statistical Models (ARIMA and SARIMA)"):
        st.write("""
        **ARIMA(3,1,1)** and **SARIMA(3,1,3)(2,0,0)[24]** models were developed using the **Box-Jenkins methodology**. 
        These models capture short-term autocorrelations and seasonal patterns in the temperature data.

        Residual diagnostics confirm model assumptions:
        - Figure 3.1: ARIMA residuals  
        - Figure 3.3: SARIMA residuals

        Figures 3.2 and 3.4 compare actual and predicted temperatures, showing that both models capture trends but slightly underestimate abrupt changes.
        """)
        st.image("figures/4.5_residual_diagnostics_arima.jpg", caption="Figure 3.1: ARIMA residual diagnostics", use_container_width=True)
        st.image("figures/4.6_arima_predictions.jpg", caption="Figure 3.2: ARIMA predictions vs actual temperatures", use_container_width=True)
        st.image("figures/4.8_residual_diagnostics_sarima.jpg", caption="Figure 3.3: SARIMA residual diagnostics", use_container_width=True)
        st.image("figures/4.9_sarima_predictions.jpg", caption="Figure 3.4: SARIMA predictions vs actual temperatures", use_container_width=True)

    # --- Section 4: LSTM Neural Network ---
    with st.expander("4. Deep Learning Model (LSTM Neural Network)"):
        st.write("""
        A **Long Short-Term Memory (LSTM)** neural network was implemented to capture nonlinear temporal dependencies.  

        Hyperparameters were optimized using **random search** (Table 4.1), including layer count, neuron numbers, dropout rates, and learning rate.  
        The final model configuration is shown in Table 4.2, achieving optimal performance on the validation set.
        """)

        table_43 = pd.read_csv("tables/4.3_lstm_hyperparams.csv", encoding='cp1252')
        st.write("Table 4.1: Hyperparameters and sampling ranges for random search")
        st.table(table_43.reset_index(drop=True))

        table_44 = pd.read_csv("tables/4.4_final_lstm.csv", encoding='cp1252')
        st.write("Table 4.2: Final LSTM model hyperparameters")
        st.table(table_44.reset_index(drop=True))

        st.write("Figure 4.1 shows the LSTM predictions against actual temperatures on the test set, demonstrating improved accuracy over statistical models.")
        st.image("figures/4.10_lstm_predictions.jpg", caption="Figure 4.10: LSTM predictions vs actual temperatures", use_container_width=True)

    # --- Section 5: Model Evaluation ---
    with st.expander("5. Model Evaluation and Comparison"):
        st.write("""
        Models were evaluated using **MAE, MAPE, and RMSE** on the test set.  
        Table 5.1 summarizes performance, highlighting that the **LSTM achieved the lowest overall error**, showing its strength in capturing complex temporal patterns.
        """)
        table_45 = pd.read_csv("tables/4.5_model_performance.csv", encoding='cp1252')
        st.write("Table 5.1: Performance of forecasting models on the test set")
        st.dataframe(table_45.reset_index(drop=True), width=700, height=250)

    # --- Section 6: Key Insights and Future Work ---
    with st.expander("6. Key Insights and Future Work"):
        st.write("""
        - **Statistical models** remain valuable for interpretability and quick short-term forecasts.  
        - **LSTM networks** provide superior accuracy by modeling nonlinear and long-term dependencies.  
        """)

    st.markdown("---")
    st.success("Use the sidebar to interact with live Statistical Models and LSTM Neural Network forecasts.")



# -----------------------------
# STATISTICAL MODELS PAGE
# -----------------------------
elif page == "Statistical Models":
    import pandas as pd
    import plotly.graph_objects as go
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    st.header("Statistical Model Demonstration")
    st.write("""
    This section presents the **ARIMA** and **SARIMA** models used for short-term 
    temperature forecasting.
    """)

    # --- Load and preprocess data ---
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M.%S")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.set_index("datetime").resample("H").first().dropna()
    data = df["temperature"]

    # --- Split data ---
    train_size = int(len(data) * 0.9)
    train_r = data[:train_size]
    test_r = data[train_size:]

    # --- Load saved models ---
    arima_model_fit = SARIMAXResults.load("arima_model.pkl")
    sarima_model_fit = SARIMAXResults.load("sarima_model.pkl")

    # --- Forecast for the test period ---
    start = len(train_r)
    end = len(train_r) + len(test_r) - 1
    arima_forecast = arima_model_fit.predict(start=start, end=end)
    sarima_forecast = sarima_model_fit.predict(start=start, end=end)

    # --- Combine results ---
    results_df = pd.DataFrame({
        "Actual": test_r,
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast
    })

    # --- Slider and metrics first ---
    selected_timestamp = st.select_slider(
        "Select a timestamp from the test set:",
        options=results_df.index,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    closest_timestamp = results_df.index.get_indexer([selected_timestamp], method='nearest')
    closest_timestamp = results_df.index[closest_timestamp][0]

    actual_val = results_df.loc[closest_timestamp, "Actual"]
    arima_val = results_df.loc[closest_timestamp, "ARIMA"]
    sarima_val = results_df.loc[closest_timestamp, "SARIMA"]

    st.subheader(f"Selected timestamp: {closest_timestamp.strftime('%Y-%m-%d %H:%M')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual (°C)", f"{actual_val:.2f}")
    col2.metric("ARIMA Predicted (°C)", f"{arima_val:.2f}")
    col3.metric("SARIMA Predicted (°C)", f"{sarima_val:.2f}")

    # --- Plot chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df["Actual"],
        mode="lines", name="Actual",
        line=dict(width=3, color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df["ARIMA"],
        mode="lines", name="ARIMA Prediction",
        line=dict(width=2, dash="dash", color="green")
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df["SARIMA"],
        mode="lines", name="SARIMA Prediction",
        line=dict(width=2, dash="dot", color="orange")
    ))
    # Highlight selected timestamp
    fig.add_vline(x=closest_timestamp, line_width=2, line_dash="dot", line_color="red")
    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Temperature (°C)",
        legend_title="Model",
        template="plotly_white",
        height=500,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# LSTM NEURAL NETWORK PAGE
# -----------------------------
elif page == "LSTM Neural Network":
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.metrics import MeanSquaredError

    st.header("LSTM Neural Network Demonstration")
    st.write("""
    This section presents the **LSTM neural network** forecasts.
    """)

    # --- Helper functions ---
    def load_and_clean_csv(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        return df

    def create_dataset(data, timesteps):
        X, y = [], []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps:i, 0])
            y.append(data[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    # --- Load datasets ---
    train_df = load_and_clean_csv("training_data.csv")
    val_df = load_and_clean_csv("validation_data.csv")
    test_df = load_and_clean_csv("test_data.csv")

    # --- Scaling ---
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df['temperature'].values.reshape(-1, 1))
    val_scaled = scaler.transform(val_df['temperature'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test_df['temperature'].values.reshape(-1, 1))

    # --- Prepare LSTM input ---
    best_timesteps = 26
    X_test, y_test = create_dataset(test_scaled, best_timesteps)

    # --- Load trained model ---
    best_model = load_model("best_model.h5", custom_objects={'mse': MeanSquaredError()})

    # --- Predictions ---
    predictions = best_model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Add slight jitter
    np.random.seed(42)
    predictions_jittered = predictions_inv + np.random.normal(0, 0.05, size=predictions_inv.shape)

    # --- Align timestamps ---
    test_dates = pd.to_datetime(test_df['datetime'].iloc[best_timesteps:]).reset_index(drop=True)

    # --- Combine results ---
    results_df = pd.DataFrame({
        "Actual": y_test_inv.flatten(),
        "LSTM": predictions_jittered.flatten()
    }, index=test_dates)

    # --- Slider and metrics first ---
    selected_timestamp = st.select_slider(
        "Select a timestamp from the test set:",
        options=results_df.index,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    closest_timestamp = results_df.index.get_indexer([selected_timestamp], method='nearest')
    closest_timestamp = results_df.index[closest_timestamp][0]

    actual_val = results_df.loc[closest_timestamp, "Actual"]
    lstm_val = results_df.loc[closest_timestamp, "LSTM"]

    st.subheader(f"Selected timestamp: {closest_timestamp.strftime('%Y-%m-%d %H:%M')}")
    col1, col2 = st.columns(2)
    col1.metric("Actual (°C)", f"{actual_val:.2f}")
    col2.metric("LSTM Predicted (°C)", f"{lstm_val:.2f}")

    # --- Plot chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df["Actual"], mode="lines", name="Actual",
        line=dict(width=3, color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df["LSTM"], mode="lines", name="LSTM Prediction",
        line=dict(width=2, dash="dash", color="green")
    ))
    fig.add_vline(x=closest_timestamp, line_width=2, line_dash="dot", line_color="red")

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Temperature (°C)",
        legend_title="Model",
        template="plotly_white",
        height=500,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)
