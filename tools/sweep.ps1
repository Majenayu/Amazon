$ErrorActionPreference = "Continue"
$root = "D:\Vectorflow\cray\Crazy shi\Kiro WEEK\Amazon"
Set-Location $root
$log  = Join-Path $root "logs\sweep.log"
"=== sweep started $(Get-Date -Format 'HH:mm:ss') ===" | Out-File $log -Encoding utf8

function Run-Cfg($name, $extra) {
    $args = @('-u','src/baseline/v3block.py','--split','train','--limit','150000',
              '--gt','data/train/train_ground_truth.tsv','--score-only','--out',
              "data/cache/v3_$name.tsv") + $extra
    "--- $name  $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log -Encoding utf8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process python -ArgumentList $args -NoNewWindow -PassThru `
         -RedirectStandardOutput "logs/${name}.out" -RedirectStandardError "logs/${name}.log"
    $p.WaitForExit()
    "    wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s exit=$($p.ExitCode)" | Add-Content $log
    Get-Content "logs/$name.log" -ErrorAction SilentlyContinue | Select-Object -Last 4 | Add-Content $log
    $a = python -u src/baseline/analyze_score.py --scores "data/cache/v3_$name.tsv" `
         --gt data/train/train_ground_truth.tsv --s1-limit 2206821 --limit 150000 `
         --other-rows 300000 2>&1
    $a | Select-Object -First 9 | Add-Content $log
    "" | Add-Content $log
}

Run-Cfg "sweepA" @('--gate','0','--budget','100000','--collect','40',
                    '--pre-collect','40','--post-budget','400','--tok-cap','600','--pre-cap','600')
Run-Cfg "sweepB" @('--gate','3','--budget','25','--collect','40',
                    '--pre-collect','40','--post-budget','400','--tok-cap','600','--pre-cap','600')
Run-Cfg "sweepC" @('--gate','3','--budget','40','--collect','60',
                    '--pre-collect','60','--post-budget','800','--tok-cap','800','--pre-cap','800')

"=== sweep finished $(Get-Date -Format 'HH:mm:ss') ===" | Add-Content $log
