@echo off
chcp 65001 >nul
echo ============================================================
echo  Attack 3: Fluency Optimization (White-Box) - Table 9 ^& 10
echo  对应论文 Section 6.3
echo  包含两个变体: Soft和Hard去混淆
echo ============================================================
echo.

set RESULTS_DIR=precomputed_results/soft_pirate_truthfulqa_full

echo ========== 变体A: Soft Prompt去混淆 (Table 9) ==========
echo.
echo [1/6] 运行Soft Fluency去混淆攻击...
python fluency_deobfuscation.py ^
    --results_dir "%RESULTS_DIR%" ^
    --embedding_file "%RESULTS_DIR%/best_candidate.pt" ^
    --deobfuscation_method soft ^
    --deobfuscated_sys_prompts_filename "deobfuscated_sys_prompt_list_soft.pt" ^
    --optimizer_iter 5 ^
    --output_token_count 15 ^
    --dataset_size 800 ^
    --batch_size 4 ^
    --window_size 3

if errorlevel 1 (
    echo [ERROR] Soft去混淆失败
    pause
    exit /b 1
)

echo.
echo [2/6] 评估Soft去混淆结果...
python evaluate_fluency_deobfuscation.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_list_file "%RESULTS_DIR%/deobfuscated_sys_prompt_list_soft.pt" ^
    --best_candidate_filename "best_deobf_sys_prompt_soft.pt" ^
    --best_candidate_scores_filename "best_deobf_sys_prompt_soft_scores.json"

echo.
echo ========== 变体B: Hard Prompt去混淆 (Table 10) ==========
echo.
echo [3/6] 运行Hard Fluency去混淆攻击（GCG优化）...
python fluency_deobfuscation.py ^
    --results_dir "%RESULTS_DIR%" ^
    --embedding_file "%RESULTS_DIR%/best_candidate.pt" ^
    --deobfuscation_method hard ^
    --deobfuscated_sys_prompts_filename "deobfuscated_sys_prompt_list_hard.pt" ^
    --optimizer_iter 5 ^
    --output_token_count 15 ^
    --dataset_size 800 ^
    --batch_size 4 ^
    --window_size 3

if errorlevel 1 (
    echo [ERROR] Hard去混淆失败
    pause
    exit /b 1
)

echo.
echo [4/6] 评估Hard去混淆结果...
python evaluate_fluency_deobfuscation.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_list_file "%RESULTS_DIR%/deobfuscated_sys_prompt_list_hard.pt" ^
    --best_candidate_filename "best_deobf_sys_prompt_hard.pt" ^
    --best_candidate_scores_filename "best_deobf_sys_prompt_hard_scores.json"

echo.
echo [5/6] 比较原始prompt vs 随机基线...
python compare_sys_prompts.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_1_conventional ^
    --sys_prompt_2_random ^
    --output_dir "%RESULTS_DIR%" ^
    --scores_filename "random_sys_prompt_scores.json" ^
    --seed 43

echo.
echo ============================================================
echo  Attack 3 完成!
echo  结果文件:
echo    Soft去混淆分数: %RESULTS_DIR%/best_deobf_sys_prompt_soft_scores.json
echo    Hard去混淆分数: %RESULTS_DIR%/best_deobf_sys_prompt_hard_scores.json
echo    随机基线分数:   %RESULTS_DIR%/random_sys_prompt_scores.json
echo  预期结果:
echo    - Soft变体: 与随机基线相当，无法恢复prompt (Table 9)
echo    - Hard变体: 可恢复个别关键词但语义不连贯 (Table 10)
echo ============================================================
pause
