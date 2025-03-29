import os
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from io import StringIO

from main.ml.model import MlModel
from main.config.logger import global_logger

from PIL import Image

from main.utils.column_utils import get_structure, Features
from main.utils.data.dto.Feature import DependentFeature
from main.utils.data.dto.MlRowValue import MlRowValue

param_query = st.query_params.get("query", None)
tester = MlModel.default().get_tester()

def process_data_csv(csv_files,callback):
    if not csv_files:
        st.error("Please upload CSV files before processing.")
        return
    df_to_process = pd.concat(csv_files) if len(csv_files) > 1 else csv_files[0]
    return process_data(df_to_process=df_to_process,callback=callback, output_csv='.temp/pppredict_data_test.csv')

def process_data_text(text,callback):
    if not text:
        st.error("Text is Required")
        return
    data = {
        'Description': [text],
        'Category': [""],
        'Sub_Category': [""]
    }

    df_to_process = pd.DataFrame(data)
    return process_data(df_to_process=df_to_process,callback=callback)

def process_data(df_to_process,callback, output_csv = None):

    try:

        df, output_csv = tester.predict(df=df_to_process, callback=callback, output_csv=output_csv)

        st.success("Data processed successfully!")

        st.write(df.head())
        if output_csv:
            st.write(f"saved to {output_csv}")
        return output_csv

    except Exception as e:

        st.error(f"An error occurred during processing: {str(e)}")
        raise e



def recursive_features_render(temp_o:MlRowValue, features:list[DependentFeature], depth = 0):
    if not temp_o or len(features or []) == 0:
        return



    st.title(f"{temp_o.name} model")
    model_path = Path(temp_o.data_dict['model_path'])
    view_dict = {
        "Model exists": model_path.exists(),
        "Model modified on": 'None'
    }
    if model_path.exists():
        view_dict["Model modified on"] = datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    st.write(view_dict)
    accuracy_validation_img_path = temp_o.data_dict['t_vs_a']

    if os.path.exists(accuracy_validation_img_path):
        image = Image.open(accuracy_validation_img_path)
        st.header("Training vs Validation")
        st.image(image, caption=accuracy_validation_img_path, use_container_width=True)

    heatmap_img_path = temp_o.data_dict['heat_map']

    if os.path.exists(heatmap_img_path):
        image = Image.open(heatmap_img_path)
        st.header("Heat map")
        st.image(image, caption=heatmap_img_path, use_container_width=True)

    if depth == 0:
        if st.radio("Show children models", [ "No", "Yes"]) == "No":
            return


    for feature in features:
        if not len(feature.children or []):
            continue
        st.write(f"{feature.name} subs")

        g = temp_o.features_dict[feature.name]
        labels = g.sub_feature_labels
        sub_tabs = st.tabs(labels)

        for label_index,label in enumerate(labels):
            with sub_tabs[label_index]:
                recursive_features_render(g.sub_feature_dict[label], feature.children, depth + 1)


def render_side():
    st.title("Model Params")

    with st.expander("Training Analysis: Section 1", True):
        recursive_features_render(tester.o, [Features.CATEGORY], 0)

    with st.expander("Categories: Section 2", False):
        st.write(get_structure())

    with st.expander("Logs: Section 3", False):
        st.code(global_logger.read_last_log_line("model_type:"), language="log")

# Streamlit UI
def main():

    st.title("Test Model")
    csv_files = []
    st.session_state.text_input = st.query_params.get("text", None)
    st.session_state.text_input  = st.text_input("Enter Transaction Title for Prediction", st.session_state.text_input )
    if st.session_state.text_input:
        st.query_params.update({"text": st.session_state.text_input})
    uploaded_files = st.file_uploader("Choose CSV Files", accept_multiple_files=True, type='csv')

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(file_content)
            csv_files.append(df)
            st.write(f"Uploaded {uploaded_file.name}")
            st.write(df.head())




    if st.button("Process Data"):
        progress_bar = st.progress(0)
        def callback(percent):
            progress_bar.progress(percent)
        if len(csv_files) > 0:
            process_data_csv(csv_files=csv_files,callback=callback)
        if st.session_state.text_input:
            process_data_text(text=st.session_state.text_input,callback=callback)

    with  st.sidebar:
        render_side()



if __name__ == "__main__":
    main()
