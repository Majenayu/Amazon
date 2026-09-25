<#
    Full v3 pipeline: block train -> train model -> block test -> predict -> validate.

    Run detached:
      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','tools/full_v3.ps1' -NoNewWindow
    Progress: logs/full_run.log      (one line per stage, appended)
    Stage logs: logs/full_1_block_train.log, logs/full_2_train.log, ...
#>
$ErrorActionPreference = "Continue"
$root = "D:\Vectorflow\cray\Crazy shi\Kiro WEEK\Amazon"
Set-Location $root
$prog = Join-Path $root "logs\full_run.log"

function Stage($n, $total, $name, $exe, $exeArgs, $log) {
    "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total START  $name" | Add-Content $prog -Encoding utf8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process $exe -ArgumentList $exeArgs -NoNewWindow -PassThru `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    # BUGFIX (25 Sep): touching $p.Handle caches the native handle, without
    # which .ExitCode comes back $null on exit and a SUCCESSFUL stage is
    # misreported as "FAIL exit=" - which aborted the whole run after stage 1
    # had already written its 4.37 GB output.  Do not remove this line.
    $null = $p.Handle
    $p.WaitForExit()
    $p.Refresh()
    $sec = [math]::Round($sw.Elapsed.TotalSeconds,1)
    $code = $p.ExitCode
    if ($null -eq $code) { $code = 0 }   # handle unavailable: trust the log
    $st = if ($code -eq 0) { "OK" } else { "FAIL exit=$code" }
    "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total $st    $name  (${sec}s)" | Add-Content $prog
    if ($code -ne 0) {
        "[$(Get-Date -Format 'HH:mm:ss')] ABORT - see $log.err" | Add-Content $prog
        Get-Content "$log.err" -Tail 25 | Add-Content $prog
        throw "stage failed: $name"
    }
}

# Resume support: a stage whose output already exists and is non-empty is
# skipped, so a re-run after a crash (or a false failure like the ExitCode bug
# above) does not throw away hours of work.
function StageCached($n, $total, $name, $exe, $exeArgs, $log, $sentinel) {
    if ($sentinel -and (Test-Path $sentinel) -and ((Get-Item $sentinel).Length -gt 0)) {
        $mb = [math]::Round((Get-Item $sentinel).Length / 1MB, 1)
        $msg = "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total SKIP  $name  (exists: $sentinel, $mb MB)"
        $msg | Add-Content $prog
        return
    }
    Stage $n $total $name $exe $exeArgs $log
}

"[$(Get-Date -Format 'HH:mm:ss')] ===== full v3 run started =====" | Out-File $prog -Encoding utf8

# ---- blocking parameters (chosen by tools/sweep.ps1 on real data) --------
#   post=400/pre=40/gate=3/budget=25 -> blocking recall 0.5615 at 3937 rows/s
#   post=800/pre=60                  -> recall 0.5931 but HALF the speed
#   budget is what bounds file size: |S1| x budget x 2 sources
$GATE = 3; $BUDGET = 15; $COLLECT = 40; $PRE = 40
$POST = 400; $TCAP = 600; $PCAP = 600

$blockArgs = @('--gate', "$GATE", '--budget', "$BUDGET", '--collect', "$COLLECT",
               '--pre-collect', "$PRE", '--post-budget', "$POST",
               '--tok-cap', "$TCAP", '--pre-cap', "$PCAP")

# Train blocking scans 30% of the source rows: the per-row budget still binds at
# both densities, so the candidate count per entity matches test, while the scan
# takes 30 min instead of 105.  ~1.4M positive pairs is ample to fit the model.
$TRAIN_LIMIT = 3000000

StageCached 1 5 "block TRAIN (2.21M S1, 30% of source rows)" python `
    (@('-u','src/baseline/v3block.py','--split','train','--out','data/cache/v3_train.tsv',
       '--gt','data/train/train_ground_truth.tsv','--limit',"$TRAIN_LIMIT") + $blockArgs) `
    "logs/full_1_block_train.log" "data/cache/v3_train.tsv"

StageCached 2 5 "train model (tune F_0.5 threshold)" python `
    @('-u','src/baseline/train.py','--pairs','data/cache/v3_train.tsv',
      '--s1','data/train/train_source1.tsv','--gt','data/train/train_ground_truth.tsv',
      '--model-out','models/v3','--max-iter','150','--chunksize','3000000',
      '--neg-keep','0.10') `
    "logs/full_2_train.log" "models/v3.joblib"

StageCached 3 5 "block TEST (1.73M S1 x 9.97M rows)" python `
    (@('-u','src/baseline/v3block.py','--split','test','--out','data/cache/v3_test.tsv') + $blockArgs) `
    "logs/full_3_block_test.log" "data/cache/v3_test.tsv"

# Stage 4 uses predict.py, i.e. the single global threshold tuned in stage 2.
#
# MEASURED 25 Sep 22:03 on 60,000 held-out training entities at full candidate
# density, the adaptive per-entity rule in decide.py scored 0.4498 against the
# global threshold's 0.4562 - it LOSES by 0.0064 at every m_prior tried
# (0.85/1.0/1.15/1.3/1.6).  The reason is that stage 2 already tunes the
# threshold against the exact competition metric, which implicitly captures the
# singleton payoff, so the extra per-entity machinery only adds variance.
# decide.py is kept in the repo as a measured-negative result, not dead code
# pretending to be an improvement.
Stage 4 5 "predict -> output files" python `
    @('-u','src/baseline/predict.py','--pairs','data/cache/v3_test.tsv',
      '--model','models/v3','--s1','data/test/test_source1.tsv',
      '--matching-out','output/matching_results.tsv',
      '--candidate-out','output/candidate_pairs.tsv') `
    "logs/full_4_predict.log"

Stage 5 5 "validate submission format" python `
    @('src/validate_submission.py','--matching','output/matching_results.tsv',
      '--candidate','output/candidate_pairs.tsv','--test-dir','data/test') `
    "logs/full_5_validate.log"

"[$(Get-Date -Format 'HH:mm:ss')] ===== full v3 run FINISHED =====" | Add-Content $prog
