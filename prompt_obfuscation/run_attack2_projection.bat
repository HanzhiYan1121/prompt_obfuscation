@echo off
chcp 65001 >nul
echo ============================================================
echo  Attack 2: Token Space Projection (White-Box) - Table 8
echo  对应论文 Section 6.2
echo ============================================================
echo.

set RESULTS_DIR=precomputed_results/soft_pirate_truthfulqa_full

echo [1/5] 使用欧氏距离投影soft prompt到token空间...
python projection.py ^
    --results_dir "%RESULTS_DIR%" ^
    --embedding_file "%RESULTS_DIR%/best_candidate.pt" ^
    --euclidean ^
    --projected_ids_filename "best_candidate_euclidean_projection.pt"

if errorlevel 1 (
    echo [ERROR] 欧氏距离投影失败
    pause
    exit /b 1
)

echo.
echo [2/5] 使用余弦相似度投影soft prompt到token空间...
python projection.py ^
    --results_dir "%RESULTS_DIR%" ^
    --embedding_file "%RESULTS_DIR%/best_candidate.pt" ^
    --cosine ^
    --projected_ids_filename "best_candidate_cosine_projection.pt"

if errorlevel 1 (
    echo [ERROR] 余弦相似度投影失败
    pause
    exit /b 1
)

echo.
echo [3/5] 比较原始prompt vs 欧氏距离投影...
python compare_sys_prompts.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_1_conventional ^
    --sys_prompt_2_file "%RESULTS_DIR%/best_candidate_euclidean_projection.pt" ^
    --output_dir "%RESULTS_DIR%" ^
    --scores_filename "euclidean_sys_prompt_scores.json"

echo.
echo [4/5] 比较原始prompt vs 余弦相似度投影...
python compare_sys_prompts.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_1_conventional ^
    --sys_prompt_2_file "%RESULTS_DIR%/best_candidate_cosine_projection.pt" ^
    --output_dir "%RESULTS_DIR%" ^
    --scores_filename "cosine_sys_prompt_scores.json"

echo.
echo [5/5] 比较原始prompt vs 随机基线...
python compare_sys_prompts.py ^
    --results_dir "%RESULTS_DIR%" ^
    --sys_prompt_1_conventional ^
    --sys_prompt_2_random ^
    --output_dir "%RESULTS_DIR%" ^
    --scores_filename "random_sys_prompt_scores.json" ^
    --seed 43

echo.
echo ============================================================
echo  Attack 2 完成!
echo  结果文件:
echo    欧氏距离投影分数: %RESULTS_DIR%/euclidean_sys_prompt_scores.json
echo    余弦投影分数:     %RESULTS_DIR%/cosine_sys_prompt_scores.json
echo    随机基线分数:     %RESULTS_DIR%/random_sys_prompt_scores.json
echo  预期结果: 投影结果应与随机基线相当，无法恢复有意义文本
echo ============================================================
pause
