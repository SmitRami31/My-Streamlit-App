import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configure page
st.set_page_config(page_title="EDA App", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar section
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("Please upload a CSV file")
    
    st.header("EDA Options")
    eda_options = st.multiselect(
        "Select EDA tasks to perform:",
        options=[
            "Show Dataset",
            "Show Tail",
            "Data Statistics",
            "Missing Values",
            "Data Types",
            "Unique Values",
            "Correlation Matrix",
            "Interactive Visualization",
            "Value Counts Bar Chart"
        ]
    )

# Main dashboard
st.title("Exploratory Data Analysis Dashboard")

if st.session_state.df is not None:
    df = st.session_state.df
    
    # Show selected EDA outputs
    if "Show Dataset" in eda_options:
        st.subheader("Row Data")
        st.dataframe(df)

    if "Show Tail" in eda_options:
        st.subheader("Last 5 Rows")
        st.dataframe(df.tail())

    if "Data Statistics" in eda_options:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

    if "Missing Values" in eda_options:
        st.subheader("Missing Values Analysis")
        
        # Show missing values summary
        missing = df.isnull().sum().to_frame("Missing Values")
        missing["Percentage"] = round(missing["Missing Values"] / len(df) * 100, 2)
        st.dataframe(missing)
    
        # Missing value handling section
        st.markdown("---")
        st.subheader("Handle Missing Values")
        
        # Select handling method
        handling_method = st.selectbox(
            "Select handling method:",
            options=[
                "Select method",
                "Drop rows with missing values",
                "Drop columns with missing values",
                "Fill with mean (numerical only)",
                "Fill with median (numerical only)",
                "Fill with mode (categorical only)",
                "Fill with specific value"
            ]
        )

        # Missing value handling
        st.subheader("Missing Value Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing_data, 'missing_percent': missing_percent})
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percent', ascending=False)
        
        if not missing_df.empty:
            st.write("Missing values summary:")
            st.dataframe(missing_df)
        else:
            st.write("No Missing values to be found!")

        # Method implementation
        if handling_method != "Select method":
            cols_with_missing = missing[missing["Missing Values"] > 0].index.tolist()
            selected_cols = st.multiselect("Select columns to handle:", cols_with_missing)
            
            if selected_cols:
                if handling_method == "Drop rows with missing values":
                    if st.button("Apply Row Deletion"):
                        st.session_state.df = df.dropna(subset=selected_cols)
                        st.success(f"Dropped rows with missing values in {selected_cols}")
                        
                elif handling_method == "Drop columns with missing values":
                    if st.button("Apply Column Deletion"):
                        st.session_state.df = df.drop(columns=selected_cols)
                        st.success(f"Dropped columns: {selected_cols}")
                        
                elif handling_method.startswith("Fill with"):
                    if "specific value" in handling_method:
                        fill_value = st.text_input("Enter fill value:")
                    else:
                        fill_value = None
                    
                    if st.button("Apply Imputation"):
                        for col in selected_cols:
                            if handling_method == "Fill with mean (numerical only)":
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    st.session_state.df[col] = df[col].fillna(df[col].mean())
                                else:
                                    st.warning(f"Column {col} is not numerical. Mean imputation skipped.")
                            elif handling_method == "Fill with median (numerical only)":
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    st.session_state.df[col] = df[col].fillna(df[col].median())
                                else:
                                    st.warning(f"Column {col} is not numerical. Median imputation skipped.")
                            elif handling_method == "Fill with mode (categorical only)":
                                if pd.api.types.is_string_dtype(df[col]):
                                    st.session_state.df[col] = df[col].fillna(df[col].mode()[0])
                                else:
                                    st.warning(f"Column {col} is not categorical. Mode imputation skipped.")
                            elif handling_method == "Fill with specific value":
                                try:
                                    # Try to convert value to appropriate type
                                    converted_value = pd.to_numeric(fill_value, errors='ignore')
                                    st.session_state.df[col] = df[col].fillna(converted_value)
                                except:
                                    st.error("Invalid fill value format")
                                    break
                        st.success("Imputation applied successfully!")
        
        # Show updated missing values summary after handling
        st.markdown("---")
        st.subheader("Updated Missing Values Summary")
        updated_missing = st.session_state.df.isnull().sum().to_frame("Missing Values")
        updated_missing["Percentage"] = round(updated_missing["Missing Values"] / len(st.session_state.df) * 100, 2)
        st.dataframe(updated_missing)

    if "Data Types" in eda_options:
        st.subheader("Data Types Summary")
        dtypes = df.dtypes.to_frame("Data Type")
        st.dataframe(dtypes)

    if "Unique Values" in eda_options:
        st.subheader("Unique Values Count")
        unique = df.nunique().to_frame("Unique Values")
        st.dataframe(unique)

    if "Correlation Matrix" in eda_options:
        st.subheader("Correlation Matrix")
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig)

    if "Interactive Visualization" in eda_options:
        st.subheader("Interactive Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin"]
            )
        
        with col2:
            x_axis = st.selectbox("X-axis", df.columns)
            y_axis = st.selectbox("Y-axis", df.columns) if chart_type not in ["Histogram", "Box", "Violin"] else None
        
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis)
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_axis)
        st.plotly_chart(fig)

    if "Value Counts Bar Chart" in eda_options:
        st.subheader("Value Counts Visualization")
        
        # Select columns for value counts
        selected_columns = st.multiselect(
            "Select columns to visualize value counts:",
            options=df.columns,
            key="value_counts_columns"
        )
        
        if selected_columns:
            # Add transformation options
            st.markdown("**Transformation Options**")
            col1, col2 = st.columns(2)
            with col1:
                use_log_scale = st.checkbox("Apply Log Scale to Y-axis", key="log_scale")
            with col2:
                compare_columns = st.checkbox("Compare Columns Side-by-Side", key="compare_columns")
            
            if compare_columns and len(selected_columns) > 1:
                # Show side-by-side bar charts
                st.markdown("**Comparison Mode**")
                for col in selected_columns:
                    value_counts = df[col].value_counts(dropna=False).reset_index()
                    value_counts.columns = ['Value', 'Count']
                    total = value_counts['Count'].sum()
                    value_counts['Percentage'] = (value_counts['Count'] / total * 100).round(2)
                    
                    # Create interactive bar chart
                    fig = px.bar(
                        value_counts,
                        x='Value',
                        y='Count',
                        text='Percentage',
                        color='Value',
                        labels={'Count': 'Number of Occurrences'},
                        height=500,
                        log_y=use_log_scale,
                        title=f"Value Counts for {col}"
                    )
                    fig.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside',
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5
                    )
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title="Count (Log Scale)" if use_log_scale else "Count",
                        showlegend=False,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data toggle
                    if st.checkbox(f"Show raw value counts for {col}", key=f"raw_{col}"):
                        st.dataframe(value_counts)
            
            else:
                # Show combined bar chart
                st.markdown("**Single Column Mode**")
                combined_data = pd.DataFrame()
                for col in selected_columns:
                    value_counts = df[col].value_counts(dropna=False).reset_index()
                    value_counts.columns = ['Value', 'Count']
                    value_counts['Column'] = col
                    combined_data = pd.concat([combined_data, value_counts])
                
                # Create interactive bar chart
                fig = px.bar(
                    combined_data,
                    x='Value',
                    y='Count',
                    color='Column',
                    barmode='group', 
                    text='Count',
                    labels={'Count': 'Number of Occurrences'},
                    height=500,
                    log_y=use_log_scale,
                    title="Value Counts Comparison"
                )
                fig.update_traces(
                    textposition='outside',
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5
                )
                fig.update_layout(
                    xaxis_title="Value",
                    yaxis_title="Count (Log Scale)" if use_log_scale else "Count",
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data toggle
                if st.checkbox("Show raw value counts for all columns", key="raw_combined"):
                    st.dataframe(combined_data)

    # Data Summary section
    st.subheader("Data Summary")
    summary = pd.DataFrame({
        "Metric": [
            "Number of Rows", "Number of Columns", 
            "Missing Values", "Duplicate Rows",
            "Numerical Columns", "Categorical Columns"
        ],
        "Value": [
            df.shape[0], df.shape[1],
            df.isnull().sum().sum(),
            df.duplicated().sum(),
            len(df.select_dtypes(include=np.number).columns),
            len(df.select_dtypes(include="object").columns)
        ]
    })
    st.dataframe(summary, hide_index=True)

    # Save modified data
    st.subheader("Save Modified Data")
    if st.button("Download CSV"):
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="modified_data.csv",
            mime="text/csv"
        )

else:
    st.warning("Please upload a CSV file to begin analysis")