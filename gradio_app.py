import gradio as gr
from test_protbert import run_inference

with gr.Blocks(title="Codon Optimization PPLM Tool") as demo:
    gr.Markdown("# 🧬 Codon Optimization Tool")
    gr.Markdown("This tool uses a fine-tuned ProtBERT model to optimize codon sequences for expression in a host organism.")
    with gr.Row():
        with gr.Column():
            mask = gr.Checkbox(label="Use Codon Masking", value=True)
            stability_type = gr.Dropdown(['mfe'], value='mfe', label="Stability Type")
            tool_pkg = gr.Dropdown(['vienna'], value='vienna', label="Tool Package")
            temperature = gr.Slider(25, 45, value=37, step=1, label="Temperature (°C)")
            host_organism = gr.Dropdown(['human', 'ecoli', 'cho'], value='human', label="Host Organism")
            protein_seq = gr.Dropdown(['sars_cov2', 'vzv', 'cho', 'ecoli', 'human' ],label=" Custom Protein Sequence, Species Specific Test Dataset or Vaccine", value="sars_cov2")
            model_type = gr.Dropdown(['human-random', 'human-long', 'human-short', 'ecoli', 'cho'], value='human-long', label="Model Type")
            predict_btn = gr.Button("🚀 Predict")

        with gr.Column():
            orf_output = gr.Textbox(label="Optimized ORF", placeholder="Optimized ORF will be displayed here")
            cai_output = gr.Number(label="CAI Optimized ORF")
            cai_orig_output = gr.Number(label="CAI Wild Type")
            stb_output = gr.Number(label="Stability of Optimized ORF")
            stb_orig_output = gr.Number(label="Stability of Wild Type")
            gc_output = gr.Number(label="GC Content of Optimized ORF")
            gc_orig_output = gr.Number(label="GC Content of Wild Type")

    predict_btn.click(
        fn=run_inference,
        inputs=[mask, stability_type, tool_pkg, temperature, host_organism, protein_seq, model_type],
        outputs=[orf_output, cai_output, cai_orig_output, stb_output, stb_orig_output, gc_output, gc_orig_output],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True)