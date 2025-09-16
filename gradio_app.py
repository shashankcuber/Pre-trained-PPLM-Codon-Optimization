import gradio as gr
from test_protbert import run_inference  # your existing function

# Default custom AA sequence shown to the user
DEFAULT_CUSTOM_AA = "MYRRCIASRWGTAAGKKPTLSGGGRETSPARTRSSFFVF"

def infer_wrapper(mask, stability_type, tool_pkg, temperature, host_organism,
                  protein_choice, model_type, custom_seq):
    use_custom = (protein_choice == "custom")
    seq_input = (custom_seq or "").strip() if use_custom else protein_choice

    # If user somehow clears the box entirely, keep it empty and return gentle blanks
    if use_custom and not seq_input:
        return (
            "", None, gr.update(value=None, visible=False),
            None, gr.update(value=None, visible=False),
            None, gr.update(value=None, visible=False),
        )

    orf, cai_opt, cai_wt, stb_opt, stb_wt, gc_opt, gc_wt = run_inference(
        mask, stability_type, tool_pkg, temperature, host_organism, seq_input, model_type
    )

    if use_custom:
        return (
            orf,
            float(cai_opt),
            gr.update(value=None, visible=False),          # hide CAI WT
            float(stb_opt),
            gr.update(value=None, visible=False),          # hide Stability WT
            float(gc_opt),
            gr.update(value=None, visible=False),          # hide GC WT
        )
    else:
        return (
            orf,
            float(cai_opt),
            float(cai_wt),
            float(stb_opt),
            gr.update(value=float(stb_wt), visible=True),
            float(gc_opt),
            gr.update(value=float(gc_wt), visible=True),
        )

with gr.Blocks(title="Codon Optimization PPLM Tool") as demo:
    gr.Markdown("# 🧬 PPLM-CO Tool")
    # gr.Markdown("This tool uses a fine-tuned ProtBERT model to optimize codon sequences for expression in a host organism.")

    with gr.Row():
        with gr.Column():
            mask = gr.Checkbox(label="Use Codon Masking", value=True, visible=False)
            stability_type = gr.Dropdown(['mfe'], value='mfe', label="Stability Type")
            tool_pkg = gr.Dropdown(['vienna'], value='vienna', label="Tool Package")
            temperature = gr.Slider(25, 45, value=37, step=1, label="Temperature (°C)")
            host_organism = gr.Dropdown(['human', 'ecoli', 'cho'], value='human', label="Host Organism")

            protein_choice = gr.Dropdown(
                ['sars_cov2', 'vzv', 'custom'],
                label="Protein Sequence / Dataset / Vaccine",
                value="sars_cov2"
            )

            # Start hidden; value empty. We’ll populate when “custom” is selected.
            custom_seq = gr.Textbox(
                label="Paste Custom Protein Sequence (AA) or Use the Example Human Sequence",
                placeholder="e.g., MFVFLVLLPLVSSQCVNLTTRTQL...",
                lines=4,
                visible=False,
                value=""
            )

            # NOTE: your original list had 'human-long' as a value; align with your backend expectations.
            model_type = gr.Dropdown(
                ['human', 'ecoli', 'cho'],
                value='human',
                label="Model Type"
            )

            with gr.Row():
                predict_btn = gr.Button("🚀 Predict", variant="primary")
                stop_btn = gr.Button("🛑 Stop", variant="stop")

        with gr.Column():
            orf_output = gr.Textbox(label="Optimized ORF", placeholder="Optimized ORF will be displayed here")
            cai_output = gr.Number(label="CAI (Optimized ORF)")
            cai_orig_output = gr.Number(label="CAI (Wild Type)", visible=True)
            stb_output = gr.Number(label="Stability (Optimized ORF) in kcal/mol")
            stb_orig_output = gr.Number(label="Stability (Wild Type) in kcal/mol", visible=True)
            gc_output = gr.Number(label="GC Content (Optimized ORF)")
            gc_orig_output = gr.Number(label="GC Content (Wild Type)", visible=True)

    # When "custom" is chosen, show textbox AND prefill it with the default AA sequence.
    def on_protein_choice_change(choice):
        is_custom = (choice == "custom")
        return (
            gr.update(visible=is_custom, value=(DEFAULT_CUSTOM_AA if is_custom else "")),
            gr.update(visible=not is_custom),  # CAI WT
            gr.update(visible=not is_custom),  # Stability WT
            gr.update(visible=not is_custom),  # GC WT
        )

    protein_choice.change(
        fn=on_protein_choice_change,
        inputs=[protein_choice],
        outputs=[custom_seq, cai_orig_output, stb_orig_output, gc_orig_output]
    )

    # Launch prediction and keep the event handle for cancellation
    predict_event = predict_btn.click(
        fn=infer_wrapper,
        inputs=[mask, stability_type, tool_pkg, temperature, host_organism, protein_choice, model_type, custom_seq],
        outputs=[orf_output, cai_output, cai_orig_output, stb_output, stb_orig_output, gc_output, gc_orig_output]
    )

    # Stop button cancels the running (or queued) prediction and clears outputs
    def _clear_outputs():
        # Clear values; keep WT visibility consistent with current choice
        return "", None, None, None, None, None, None

    stop_btn.click(
        fn=_clear_outputs,
        inputs=[],
        outputs=[orf_output, cai_output, cai_orig_output, stb_output, stb_orig_output, gc_output, gc_orig_output],
        cancels=[predict_event]
    )

# Enable queueing (required for cancellation to work)
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True)