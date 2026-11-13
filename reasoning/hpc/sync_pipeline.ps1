# Laptop -> HPC: everything the pipeline check needs, packed into one file so there's only one upload.
#   powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_pipeline.ps1        (from the BN4601A folder)
$ErrorActionPreference = "Stop"
$repo   = Resolve-Path "$PSScriptRoot\..\.."
$host_  = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
$pack   = Join-Path $env:TEMP "pipeline_pack.tar"

Write-Host "Packing ..."
tar -cf $pack -C $repo reasoning/src reasoning/hpc reasoning/models/heads reasoning/data/processed/head_scores.parquet `
    reasoning/data/processed/cards_pool1.parquet reasoning/data/processed/pool1.parquet `
    reasoning/outputs/embeddings/cards_pool1__base4bit.npz `
    vitals/src vitals/models vision/detector/runs/fracture/weights/best.pt `
    vision/detector/data/yolo/splits.csv vision/detector/data/yolo/images/test vision/detector/data/yolo/labels/test vision/cato/hpc_run_3epochs/huggingface_adapter

Write-Host "Uploading ..."
scp -q $pack "${host_}:$remote/pipeline_pack.tar"

Write-Host "Unpacking on the HPC ..."
ssh $host_ "cd $remote && tar -xf pipeline_pack.tar && rm pipeline_pack.tar && find reasoning/hpc -name '*.sh' -o -name '*.pbs' -o -name '*.conf' | xargs sed -i 's/\r$//' && du -sh vision"
Remove-Item $pack
Write-Host "Done." -ForegroundColor Green
