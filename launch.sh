#!/bin/bash


exp_name="Walker_1"
python stage1_idm.py env_name="walker" exp_name="${exp_name}"
python stage2_bc.py env_name="walker" exp_name="${exp_name}"
python stage3_decoding.py env_name="walker" exp_name="${exp_name}"