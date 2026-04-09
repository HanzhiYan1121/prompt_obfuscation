@echo off
chcp 65001 >nul
echo ============================================================
echo  Attack 1: Prompt Injection (Black-Box) - Table 7
echo  对应论文 Section 6.1
echo ============================================================
echo.

set RESULTS_DIR=precomputed_results/soft_pirate_truthfulqa_full

echo [1/4] 对混淆后的soft prompt执行提取攻击...
python prompt_extraction.py ^
    --results_dir "%RESULTS_DIR%" ^
    --extraction_prompts_file "extraction_prompts/gpt4_generated.json" ^
    --output_filename "prompt_extraction_output_obfuscated.json" ^
    --tensor_file "%RESULTS_DIR%/best_candidate.pt" ^
    --batch_size 8

if errorlevel 1 (
    echo [ERROR] 混淆prompt提取攻击失败
    pause
    exit /b 1
)

echo.
echo [2/4] 对原始prompt执行提取攻击（作为对照）...
python prompt_extraction.py ^
    --results_dir "%RESULTS_DIR%" ^
    --extraction_prompts_file "extraction_prompts/gpt4_generated.json" ^
    --output_filename "prompt_extraction_output_conventional.json" ^
    --conventional ^
    --batch_size 8

if errorlevel 1 (
    echo [ERROR] 原始prompt提取攻击失败
    pause
    exit /b 1
)

echo.
echo [3/4] 评估混淆prompt的提取结果...
python evaluate_prompt_extraction.py ^
    --results_dir "%RESULTS_DIR%" ^
    --extraction_output_file "%RESULTS_DIR%/prompt_extraction_output_obfuscated.json" ^
    --rouge_recall_threshold 0.9 ^
    --successful_outputs_filename "successful_outputs_obfuscated.json"

echo.
echo [4/4] 评估原始prompt的提取结果...
python evaluate_prompt_extraction.py ^
    --results_dir "%RESULTS_DIR%" ^
    --extraction_output_file "%RESULTS_DIR%/prompt_extraction_output_conventional.json" ^
    --rouge_recall_threshold 0.9 ^
    --successful_outputs_filename "successful_outputs_conventional.json"

echo.
echo ============================================================
echo  Attack 1 完成!
echo  结果文件:
echo    混淆prompt提取: %RESULTS_DIR%/successful_outputs_obfuscated.json
echo    原始prompt提取: %RESULTS_DIR%/successful_outputs_conventional.json
echo  预期结果: 混淆prompt应为0次成功提取，原始prompt有多次成功提取
echo ============================================================
pause
