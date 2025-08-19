#!/bin/bash
python3 test_protbert.py \
--tool_pkg='vienna' \
--temperature=37 \
--host_organism='human' \
--protein_seq='MYRRCIASRWGTAAGKKPTLSGGGRETSPARTRSSFFVF' \
--model_type='human-long' \