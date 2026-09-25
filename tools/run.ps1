<#
    Single entry point for the entity-resolution pipeline.

        .\tools\run.ps1                 full pipeline, train AND test at MATCHED density
        .\tools\run.ps1 -Mode recall    blocking-recall probe (minutes) - run this first
        .\tools\run.ps1 -Mode sweep     blocking parameter sweep on a small window
        .\tools\run.ps1 -Mode predict   re-predict only, from existing candidate pairs

    WHY "MATCHED DENSITY" IS THE DEFAULT
    ------------------------------------
    Run 1 tuned its threshold on training pairs built from 30% of the Source-2/3
    rows, then served it on test pairs built from 100% of them:

        train  44,928,982 pairs / 2,206,821 S1 = 20.4 candidates per entity
        test   53,405,476 pairs / 1,732,544 S1 = 30.8 candidates per entity

    The threshold was therefore calibrated against an easier distribution than
    the one it met at serving time, and the leaderboard returned 0.56. Every
    extra candidate is another chance at a false merge, and a false merge on a
    true singleton costs a full 1.0, so a threshold that is safe at 20.4
    candidates is too permissive at 30.8.

    This script blocks train and test with the SAME parameters and NO --limit, so
    the tuned threshold is tuned at the density it is served at. See
    docs/WORKFLOW.md.

    Resumable: any stage whose output already exists and is non-empty is skipped,
    so a crash never discards hours of work. Re-running is always safe.
#>
param(
    [ValidateSet("full", "recall", "sweep", "predict")]
    [string]$Mode = "full",
    [int]$S1Limit = 20000,      # recall/sweep: Source-1 entities in the probe
    [int]$Limit = 150000        # recall/sweep: Source-2/3 rows in the probe
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path logs, models, output, data\cache | Out-Null
$prog = Join-Path $root "logs\run.log"

# Blocking parameters, chosen by -Mode sweep on real data (see logs/sweep.log).
# post=800 lifts recall 0.5708 -> 0.5930 but halves throughput.
$GATE = 3; $BUDGET = 15; $COLLECT = 40; $PRE = 40
$POST = 400; $TCAP = 600; $PCAP = 600
$blockArgs = @('--gate', "$GATE", '--budget', "$BUDGET", '--collect', "$COLLECT",
               '--pre-collect', "$PRE", '--post-budget', "$POST",
               '--tok-cap', "$TCAP", '--pre-cap', "$PCAP")

$trainPairs = "data/cache/train_pairs.tsv"
$testPairs = "data/cache/test_pairs.tsv"
$model = "models/v3"

function Stage($n, $total, $name, $exe, $exeArgs, $log) {
    "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total START  $name" | Add-Content $prog -Encoding utf8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process $exe -ArgumentList $exeArgs -NoNewWindow -PassThru `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    # Touch the handle so .ExitCode is populated. Without this, PowerShell
    # returns $null and a SUCCESSFUL stage is misreported as a failure.
    $null = $p.Handle
    $p.WaitForExit()
    $p.Refresh()
    $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    $code = $p.ExitCode
    if ($null -eq $code) { $code = 0 }
    $st = if ($code -eq 0) { "OK" } else { "FAIL exit=$code" }
    "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total $st    $name  (${sec}s)" | Add-Content $prog
    if ($code -ne 0) {
        "[$(Get-Date -Format 'HH:mm:ss')] ABORT - see $log.err" | Add-Content $prog
        Get-Content "$log.err" -Tail 25 | Add-Content $prog
        throw "stage failed: $name"
    }
}

function StageCached($n, $total, $name, $exe, $exeArgs, $log, $sentinel) {
    if ($sentinel -and (Test-Path $sentinel) -and ((Get-Item $sentinel).Length -gt 0)) {
        $mb = [math]::Round((Get-Item $sentinel).Length / 1MB, 1)
        $msg = "[$(Get-Date -Format 'HH:mm:ss')] STAGE $n/$total SKIP  $name  (exists, $mb MB)"
        $msg | Add-Content $prog
        return
    }
    Stage $n $total $name $exe $exeArgs $log
}

"[$(Get-Date -Format 'HH:mm:ss')] ===== run.ps1 -Mode $Mode started =====" | Add-Content $prog

# ---------------------------------------------------------------- recall probe
if ($Mode -eq "recall") {
    $probe = "data/cache/probe.tsv"
    Remove-Item $probe -Force -ErrorAction SilentlyContinue
    Stage 1 2 "probe: block $S1Limit S1 x $Limit rows (evidence score only)" python `
        (@('-u','src/baseline/block.py','--split','train','--score-only',
           '--out',$probe,'--gt','data/train/train_ground_truth.tsv',
           '--s1-limit',"$S1Limit",'--limit',"$Limit") + $blockArgs) `
        "logs\probe.log"
    Stage 2 2 "measure blocking recall" python `
        @('-u','src/baseline/measure.py','recall','--pairs',$probe,
          '--gt','data/train/train_ground_truth.tsv','--s1-limit',"$S1Limit",'--limit',"$Limit") `
        "logs\probe_recall.log"
    return
}

# ------------------------------------------------------------------ param sweep
if ($Mode -eq "sweep") {
    foreach ($cfg in @(@(400,600,3,15), @(400,600,0,40), @(800,800,3,40))) {
        $post, $cap, $gate, $budget = $cfg
        $out = "data/cache/sweep_${post}_${cap}_${gate}_${budget}.tsv"
        Remove-Item $out -Force -ErrorAction SilentlyContinue
        Stage 1 1 "sweep post=$post cap=$cap gate=$gate budget=$budget" python `
            (@('-u','src/baseline/block.py','--split','train','--score-only',
               '--out',$out,'--gt','data/train/train_ground_truth.tsv',
               '--s1-limit',"$S1Limit",'--limit',"$Limit",
               '--gate',"$gate",'--budget',"$budget",'--collect',"$COLLECT",
               '--pre-collect',"$PRE",'--post-budget',"$post",
               '--tok-cap',"$cap",'--pre-cap',"$cap")) `
            "logs\sweep_${post}_${cap}.log"
        python -u src/baseline/measure.py curve --pairs $out --gt data/train/train_ground_truth.tsv --s1-limit $S1Limit --limit $Limit 2>&1 |
            Add-Content "logs\sweep.log"
    }
    return
}

# ----------------------------------------------------------------- predict only
if ($Mode -eq "predict") {
    Stage 1 2 "predict -> output files" python `
        @('-u','src/baseline/predict.py','--pairs',$testPairs,'--model',$model,
          '--s1','data/test/test_source1.tsv',
          '--matching-out','output/matching_results.tsv',
          '--candidate-out','output/candidate_pairs.tsv') `
        "logs\predict.log"
    Stage 2 2 "validate submission format" python `
        @('src/validate_submission.py','--matching','output/matching_results.tsv',
          '--candidate','output/candidate_pairs.tsv','--test-dir','data\test') `
        "logs\validate.log"
    return
}

# ----------------------------------------------------------------- full pipeline
# No --limit anywhere: train and test must be blocked at the same density.
StageCached 1 4 "block TRAIN (2.21M S1 x 10.32M rows, full density)" python `
    (@('-u','src/baseline/block.py','--split','train','--out',$trainPairs,
       '--gt','data/train/train_ground_truth.tsv') + $blockArgs) `
    "logs\block_train.log" $trainPairs

StageCached 2 4 "train + tune threshold at matched density" python `
    @('-u','src/baseline/train.py','--pairs',$trainPairs,
      '--s1','data/train/train_source1.tsv','--gt','data/train/train_ground_truth.tsv',
      '--model-out',$model,'--max-iter','150','--chunksize','3000000',
      '--neg-keep','0.10') `
    "logs\train.log" "$model.joblib"

StageCached 3 4 "block TEST (1.73M S1 x 9.97M rows, full density)" python `
    (@('-u','src/baseline/block.py','--split','test','--out',$testPairs) + $blockArgs) `
    "logs\block_test.log" $testPairs

Stage 4 5 "predict -> output files" python `
    @('-u','src/baseline/predict.py','--pairs',$testPairs,'--model',$model,
      '--s1','data/test/test_source1.tsv',
      '--matching-out','output/matching_results.tsv',
      '--candidate-out','output/candidate_pairs.tsv') `
    "logs\predict.log"

Stage 5 5 "validate submission format" python `
    @('src/validate_submission.py','--matching','output/matching_results.tsv',
      '--candidate','output/candidate_pairs.tsv','--test-dir','data\test') `
    "logs\validate.log"

"[$(Get-Date -Format 'HH:mm:ss')] ===== run.ps1 -Mode $Mode FINISHED =====" | Add-Content $prog

