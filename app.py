import gradio as gr
import os
import mlflow
from config import cfg

# --- Import Logic from Modules ---
from data_utils import (
    load_and_store_data, split_and_save_data, find_missing_values, 
    find_categorical_columns, get_shape, generate_profile_report, 
    preprocess_data, update_task_type
)
from plot_utils import plot_task_analysis, plot_all_distributions, plot_correlation
from model_utils import (
    train_models_pipeline, tune_models_pipeline, update_model_choices
)

# --- Initialize MLflow ---
mlflow.set_tracking_uri(f"file://{os.path.abspath(cfg.EXPERIMENT_DIR)}")
mlflow.set_experiment("AutoML_App")

# --- The Gradio App ---
with gr.Blocks(title="Auto-ML Pro") as demo:
    
    # --- SHARED STATE ---
    df_state = gr.State()
    X_state = gr.State()
    y_state = gr.State()
    task_type_state = gr.State(value="Regression")
    
    # Raw Splits
    X_tr_raw = gr.State(); y_tr_raw = gr.State()
    X_val_raw = gr.State(); y_val_raw = gr.State()
    X_ts_raw = gr.State(); y_ts_raw = gr.State()
    
    # Final Processed Splits
    X_tr_fin = gr.State(); y_tr_fin = gr.State()
    X_val_fin = gr.State(); y_val_fin = gr.State()
    X_ts_fin = gr.State(); y_ts_fin = gr.State()
    
    # DataFrame for Plotting
    X_plot_state = gr.State()

    gr.Markdown("<h1 style='text-align: center;'>🚀 Auto-ML Studio</h1>")
    
    with gr.Tabs():
        
        # --- TAB 1: Upload ---
        with gr.TabItem("1. Upload & Split"):
            with gr.Row():
                f_in = gr.File(label="CSV File", file_types=[".csv"])
                upl_btn = gr.Button("Upload", variant="primary")
            df_out = gr.DataFrame(label="Data Preview")
            
            gr.Markdown("---")
            y_col = gr.Textbox(label="Target Column Name")
            split_btn = gr.Button("Split Data")
            sts = gr.Textbox(label="Status")
            
            upl_btn.click(load_and_store_data, f_in, [df_out, df_state])
            split_btn.click(split_and_save_data, [df_state, y_col], 
                            [sts, df_out, df_out, X_tr_raw, y_tr_raw, X_tr_raw, y_tr_raw, X_val_raw, y_val_raw, X_ts_raw, y_ts_raw])
            # Note: X_tr_raw repeated twice in outputs above to handle potential preview vs state mapping if needed, 
            # but based on function return: msg, preview_X, preview_y, X_train, y_train...
            # Correct mapping: [sts, preview_X, preview_y, X_tr_raw, y_tr_raw, X_val_raw, y_val_raw, X_ts_raw, y_ts_raw]

        # --- TAB 2: EDA ---
        with gr.TabItem("2. EDA"):
            with gr.Row():
                b1 = gr.Button("Check Missing")
                b2 = gr.Button("Check Categorical")
                b3 = gr.Button("Check Shape")
            
            with gr.Row():
                out1 = gr.DataFrame(label="Missing Values")
                out2 = gr.DataFrame(label="Categorical Columns")
            out3 = gr.Textbox(label="Dataset Shape")
            
            b1.click(find_missing_values, df_state, out1)
            b2.click(find_categorical_columns, df_state, out2)
            b3.click(get_shape, df_state, out3)

        # --- TAB 2.5: Profiling ---
        with gr.TabItem("2.5. Advanced Report"):
            rep_btn = gr.Button("Generate Full Report")
            rep_out = gr.HTML(label="Pandas Profiling")
            rep_btn.click(generate_profile_report, df_state, rep_out)

        # --- TAB 3: Preprocessing ---
        with gr.TabItem("3. Preprocessing"):
            rad = gr.Radio(["Regression", "Classification"], label="Task Type", value="Regression")
            rad.change(update_task_type, rad, task_type_state)
            
            proc_btn = gr.Button("Run Pipeline", variant="primary")
            proc_sts = gr.Textbox(label="Status", lines=5)
            proc_view = gr.DataFrame(label="Processed Train Data")
            
            proc_btn.click(preprocess_data, 
                           [X_tr_raw, y_tr_raw, X_val_raw, y_val_raw, X_ts_raw, y_ts_raw, task_type_state],
                           [proc_sts, proc_view, X_tr_fin, y_tr_fin, X_val_fin, y_val_fin, X_ts_fin, y_ts_fin, X_plot_state])

        # --- TAB 4: Analysis ---
        with gr.TabItem("4. Analysis"):
            ana_btn = gr.Button("Analyze Raw Data")
            with gr.Row():
                ana_plot = gr.Plot(label="Visualization")
                ana_msg = gr.Textbox(label="Analysis Report")
            ana_btn.click(plot_task_analysis, [X_state, y_state, task_type_state], [ana_plot, ana_msg])

        # --- TAB 5: Visuals ---
        with gr.TabItem("5. Model Visuals"):
            with gr.Row():
                p1_btn = gr.Button("Plot Distributions")
                p2_btn = gr.Button("Plot Correlation")
            p_out = gr.Plot(label="Plot Output")
            p1_btn.click(plot_all_distributions, X_plot_state, p_out)
            p2_btn.click(plot_correlation, X_plot_state, p_out)

        # --- TAB 6: Train ---
        with gr.TabItem("6. Train"):
            trn_btn = gr.Button("Run Auto-ML Experiment", variant="primary")
            
            with gr.Row():
                lb = gr.DataFrame(label="Leaderboard")
                dl = gr.File(label="Best Model")
            
            cp = gr.Plot(label="Comparison Chart")
            t_st = gr.Textbox(label="Summary")
            
            trn_btn.click(train_models_pipeline, 
                          [X_tr_fin, y_tr_fin, X_val_fin, y_val_fin, X_ts_fin, y_ts_fin, task_type_state],
                          [lb, cp, dl, t_st])

        # --- TAB 7: Tune ---
        with gr.TabItem("7. Tune"):
            ref_btn = gr.Button("Refresh Models for Task")
            mod_sel = gr.Dropdown(label="Select Model")
            tun_btn = gr.Button("Start Tuning", variant="primary")
            
            tun_st = gr.Textbox(label="Tuning Result", lines=5)
            with gr.Row():
                tun_pl = gr.Plot(label="Performance")
                tun_fl = gr.File(label="Tuned Model")
            
            ref_btn.click(update_model_choices, task_type_state, mod_sel)
            tun_btn.click(tune_models_pipeline,
                          [X_tr_fin, y_tr_fin, X_val_fin, y_val_fin, X_ts_fin, y_ts_fin, mod_sel, task_type_state],
                          [tun_st, tun_pl, tun_fl])

if __name__ == "__main__":
    demo.launch(share=True)